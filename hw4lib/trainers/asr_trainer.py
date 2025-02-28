from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ..decoding.sequence_generator import SequenceGenerator
from ..utils import create_scheduler
from ..model import DecoderOnlyTransformer
import torchaudio.functional as aF
import json
import torchmetrics.text as tmt

class ASRTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        
        # Initialize CE loss
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss'].get('label_smoothing', 0.0)
        )
        
        # Initialize CTC loss if needed
        self.ctc_criterion = None
        self.ctc_weight = self.config['loss'].get('ctc_weight', 0.0)
        if self.ctc_weight > 0:
            self.ctc_criterion = nn.CTCLoss(
                blank=self.tokenizer.pad_id,
                zero_infinity=True
            )

    def _train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: Training metrics and attention weights
        """
        # Initialize training variables
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Training ASR]")
        running_ce_loss = 0.0
        running_ctc_loss = 0.0
        running_joint_loss = 0.0
        total_tokens = 0
        running_att = None  # Initialize running_att here

        # Only zero gradients when starting a new accumulation cycle
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # Unpack batch
            feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths = batch
            feats = feats.to(self.device)
            targets_shifted = targets_shifted.to(self.device)
            targets_golden = targets_golden.to(self.device)
            feat_lengths = feat_lengths.to(self.device)
            transcript_lengths = transcript_lengths.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                # Forward pass
                seq_out, curr_att, ctc_inputs = self.model(
                    feats, targets_shifted, feat_lengths, transcript_lengths
                )
                running_att = curr_att  # Update running_att with the latest attention weights
                
                # Calculate CE loss
                ce_loss = self.ce_criterion(
                    seq_out.view(-1, seq_out.size(-1)),
                    targets_golden.view(-1)
                )
                
                # Calculate CTC loss if needed
                if self.ctc_weight > 0:
                    ctc_loss = self.ctc_criterion(
                        ctc_inputs['log_probs'],
                        targets_golden,
                        ctc_inputs['lengths'],
                        transcript_lengths
                    )
                    loss = ce_loss + self.ctc_weight * ctc_loss
                else:
                    ctc_loss = torch.tensor(0.0)
                    loss = ce_loss

                # Normalize loss by accumulation steps
                loss = loss / self.config['training']['gradient_accumulation_steps']

            # Backward pass
            self.scaler.scale(loss).backward()

            # Only update weights after accumulating enough gradients
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            # Calculate metrics
            batch_tokens = transcript_lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += ce_loss.item() * batch_tokens
            if self.ctc_weight > 0:
                running_ctc_loss += ctc_loss.item() * batch_tokens
            running_joint_loss += loss.item() * batch_tokens

            # Update progress bar
            avg_ce_loss = running_ce_loss / total_tokens
            avg_ctc_loss = running_ctc_loss / total_tokens
            avg_joint_loss = running_joint_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_ce_loss))
            
            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ctc_loss=f"{avg_ctc_loss:.4f}", 
                joint_loss=f"{avg_joint_loss:.4f}",
                perplexity=f"{perplexity:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Clean up
            del feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths
            del seq_out, curr_att, ctc_inputs, loss
            torch.cuda.empty_cache()

        # Handle remaining gradients
        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        # Compute final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        avg_ctc_loss = running_ctc_loss / total_tokens
        avg_joint_loss = running_joint_loss / total_tokens
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()))
        batch_bar.close()

        return {
            'ce_loss': avg_ce_loss,
            'ctc_loss': avg_ctc_loss,
            'joint_loss': avg_joint_loss,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, running_att

    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.
        
        Args:
            dataloader: DataLoader for validation data
        Returns:
            Tuple[Dict[str, float], List[Dict[str, Any]]]: Validation metrics and recognition results
        """
        results = self.recognize(dataloader)
        assert len(results) == len(dataloader.dataset)
        
        # Extract references and hypotheses
        references = [result['target'] for result in results]
        hypotheses = [result['generated'] for result in results]
        
        # Calculate metrics on full batch
        metrics = self._calculate_asr_metrics(references, hypotheses)
        
        return metrics, results
    
    def train(self, train_dataloader, val_dataloader, epochs: Optional[int] = None):
        """
        Full training loop for ASR training.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Optional[int], number of epochs to train
        """
        # Initialize learning rate scheduler if not already done
        if self.scheduler is None:
            self.scheduler = create_scheduler(
                self.optimizer,
                self.config['scheduler'],
                train_dataloader
            )

        # Training loop
        best_val_loss = float('inf')
        best_val_wer  = float('inf')
        best_val_cer  = float('inf')
        best_val_dist = float('inf')

        if epochs is None:
            epochs = self.config['training']['epochs']

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # TODO: Train for one epoch
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            
            # TODO: Validate
            val_metrics, val_results = self._validate_epoch(val_dataloader)

            # Step ReduceLROnPlateau scheduler with validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['cer'])
            
            # Log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # Save attention plots
            train_attn_keys = list(train_attn.keys())
            if train_attn_keys: 
                # Get the first self-attention and cross-attention layers
                decoder_self_keys  = [k for k in train_attn_keys if 'dec_self' in k]
                decoder_cross_keys = [k for k in train_attn_keys if 'dec_cross' in k]
                
                if decoder_self_keys:
                    # Plot first layer (layer1) if available
                    first_self_key = 'layer1_dec_self'
                    if first_self_key in train_attn:
                        self._save_attention_plot(train_attn[first_self_key][0], epoch, "decoder_self")
                
                if decoder_cross_keys:
                    # Plot first layer (layer1) if available
                    first_cross_key = 'layer1_dec_cross'
                    if first_cross_key in train_attn:
                        self._save_attention_plot(train_attn[first_cross_key][0], epoch, "decoder_cross")
            
            # Save generated text
            self._save_generated_text(val_results, f'val_epoch_{epoch}')
            
            # Save checkpoints
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            # Check if this is the best model
            if val_metrics['cer'] < best_val_cer:
                best_val_cer = val_metrics['cer']
                self.best_metric = val_metrics['cer']
                self.save_checkpoint('checkpoint-best-metric-model.pth') 
                
        # Final cleanup
        self.cleanup()

    def evaluate(self, dataloader, solution:Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on the test set.
        
        Args:
            dataloader: DataLoader for test data
            solution_json: Path to the JSON file containing the test set solutions
        Returns:
            Dictionary containing evaluation metrics and recognition results for each recognition config
        """
        if solution is not None:
            solution_data = solution
        else:
            raise ValueError("Solution is required for evaluation")

        # Get recognition configs
        recognition_configs = self._get_evaluation_recognition_configs()
        
        # Evaluate with each recognition config
        eval_results = {}
        for config_name, config in recognition_configs.items():
            print(f"Evaluating with {config_name} config")
            results = self.recognize(dataloader, config)
            assert len(results) == len(solution_data)
            
            # Calculate metrics on full batch
            metrics = self._calculate_asr_metrics(solution_data, [r['generated'] for r in results])
            eval_results[config_name] = metrics
        
        return eval_results

    def recognize(self, dataloader, recognition_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating transcriptions from audio features.
        
        Args:
            dataloader: DataLoader containing the evaluation data
            recognition_config: Optional dictionary containing recognition parameters:
                - num_batches: int, number of batches to process
                - beam_width: int, beam search width
                - temperature: float, temperature for beam search
                - repeat_penalty: float, repeat penalty for beam search
                - lm_weight: float, language model interpolation weight
                - lm_model: Optional[DecoderOnlyTransformer], language model for shallow fusion
        Returns:
            List of dictionaries containing recognition results with generated sequences and scores
            (targets included if available)
        """
        if recognition_config is None:
            # Default config (greedy search)
            recognition_config = {
                'num_batches': None,
                'beam_width': 1,
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            }

        if recognition_config.get('lm_model') is not None:
            recognition_config['lm_model'].eval()
            recognition_config['lm_model'].to(self.device)

        # Create sequence generator
        generator = SequenceGenerator(
            score_fn=None,  # Will be set for each batch
            tokenizer=self.tokenizer,
            max_length=self.model.max_len,
            device=self.device
        )

        # Initialize variables
        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Recognizing ASR]")
        results = []

        # Run inference
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # Unpack batch (handle both cases where targets might be None)
                feats, _, targets_golden, feat_lengths, _ = batch
                feats = feats.to(self.device)
                feat_lengths = feat_lengths.to(self.device)
                if targets_golden is not None:
                    targets_golden = targets_golden.to(self.device)
                
                # Encode features
                encoder_output, pad_mask_src, _, _ = self.model.encode(feats, feat_lengths)

                # Define scoring function for this batch
                def get_score(x):
                    asr_logits = self.model.score(x, encoder_output, pad_mask_src)
                    if recognition_config.get('lm_model') is not None:
                        lm_logits = recognition_config['lm_model'].score(x)
                        return asr_logits + recognition_config['lm_weight'] * lm_logits
                    return asr_logits
                
                # Set score function
                generator.score_fn = get_score

                # Generate sequences
                batch_size = feats.size(0)
                prompts = torch.full((batch_size, 1), self.tokenizer.sos_id, dtype=torch.long).to(self.device)

                # Generate sequences using beam search or greedy search
                if recognition_config['beam_width'] > 1:
                    seqs, scores = generator.generate_beam(
                        x=prompts,
                        beam_width=recognition_config['beam_width'],
                        temperature=recognition_config.get('temperature', 1.0),
                        repeat_penalty=recognition_config.get('repeat_penalty', 1.0)
                    )
                    # Pick best beam
                    seqs = seqs[:, 0, :]
                    scores = scores[:, 0]
                else:
                    seqs, scores = generator.generate_greedy(
                        x=prompts,
                        temperature=recognition_config.get('temperature', 1.0),
                        repeat_penalty=recognition_config.get('repeat_penalty', 1.0)
                    )

                # Clean up
                del feats, feat_lengths, encoder_output, pad_mask_src, prompts
                torch.cuda.empty_cache()

                # Post process sequences
                post_processed_preds = generator.post_process_sequence(seqs, self.tokenizer)
                
                # Store results
                
                if targets_golden is not None:
                    post_processed_targets = generator.post_process_sequence(targets_golden, self.tokenizer)
                    for j, (pred, target) in enumerate(zip(post_processed_preds, post_processed_targets)):
                        results.append({
                            'target': self.tokenizer.decode(target.tolist()),
                            'generated': self.tokenizer.decode(pred.tolist())[1:],
                            'score': scores[j].item()
                        })
                else:
                    for j, pred in enumerate(post_processed_preds):
                        results.append({
                            'generated': self.tokenizer.decode(pred.tolist())[1:],
                            'score': scores[j].item()
                        })

                batch_bar.update()

                if recognition_config['num_batches'] is not None and i >= recognition_config['num_batches'] - 1:
                    break

            batch_bar.close()
            return results

    def _get_evaluation_recognition_configs(self, lm_model: Optional[DecoderOnlyTransformer] = None, lm_weight: float = 0.0) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of recognition configurations for evaluation.
        
        Returns:
            Dictionary containing recognition configurations
        """

        common_config = {
            'num_batches': None,
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': lm_weight,
            'lm_model': lm_model
        }
        greedy_config = common_config.copy()
        greedy_config.update({
            'beam_width': 1,
        })

        beam_8_config = common_config.copy()
        beam_8_config.update({
            'beam_width': 8,
        })
        
        beam_16_config = common_config.copy()
        beam_16_config.update({
            'beam_width': 16,
        })
        
        beam_32_config = common_config.copy()
        beam_32_config.update({
            'beam_width': 32,
        })

        return {
            'greedy': greedy_config,
            'beam_8': beam_8_config,
            'beam_16': beam_16_config,
            'beam_32': beam_32_config
        }
        

    def _calculate_asr_metrics(self, references: Union[str, List[str]], hypotheses: Union[str, List[str]]) -> Tuple[float, float, float]:
        """
        Calculate Levenshtein distance, WER, CER for strings or lists of strings.
        
        Args:
            references: Reference string(s)
            hypotheses: Hypothesis string(s)
        Returns:
            Tuple of (word_dist, wer, cer)
        """
        # Initialize metrics
        wer_metric = tmt.WordErrorRate()
        word_edit_metric = tmt.EditDistance(reduction='mean')
        cer_metric = tmt.CharErrorRate()
        
        # Calculate metrics
        word_dist = word_edit_metric(hypotheses, references)
        wer = wer_metric(hypotheses, references)  # torchmetrics returns as decimal
        cer = cer_metric(hypotheses, references)  # torchmetrics returns as decimal

        return {
            'word_dist': word_dist.item(),
            'wer': wer.item() * 100,
            'cer': cer.item() * 100
        }
        
        
        