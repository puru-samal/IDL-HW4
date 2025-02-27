# Directories
HANDIN_DIR = handin
TEST_DIR = tests
AUTOGRADE_FILES = $(TEST_DIR)/testing_framework.py $(TEST_DIR)/test_hw4p1.py $(TEST_DIR)/test_mytorch_softmax.py $(TEST_DIR)/test_mytorch_scaled_dot_product_attention.py $(TEST_DIR)/test_mytorch_multi_head_attention.py

# Python command
# python3 for autolab
PYTHON = python3 

# Default target
default: grade

# Create autograde.tar containing all test files and dependencies
create_autograde:
	@echo "Creating autograde.tar..."
	@mkdir -p $(TEST_DIR)
	@tar --exclude='*.pyc' \
		--exclude='__pycache__' \
		--exclude='handin.tar' \
		--exclude='$(HANDIN_DIR)' \
		--exclude='mytorch' \
		--exclude='*.ipynb' \
		-czf autograde.tar $(AUTOGRADE_FILES)
	@echo "Created autograde.tar successfully"

# Extract both autograde.tar and student's handin.tar
setup:
	@echo "Setting up grading environment..."
	@if [ -d $(HANDIN_DIR) ]; then rm -rf $(HANDIN_DIR); fi
	@tar xf autograde.tar
	@tar xf handin.tar
	@if [ ! -d $(HANDIN_DIR) ]; then \
		echo "Error: handin.tar does not contain $(HANDIN_DIR) directory"; \
		exit 1; \
	fi
	@mv $(HANDIN_DIR)/* ./
	@rm -rf $(HANDIN_DIR)

# Run the tests
test: setup
	@echo "Running tests..."
	@$(PYTHON) -m $(TEST_DIR).test_hw4p1

# Grade the submission and output JSON results
grade: test

# Clean up
clean:
	@rm -rf $(HANDIN_DIR)
	@rm -f *.pyc
	@rm -rf __pycache__
	@rm -rf $(TEST_DIR)/__pycache__

.PHONY: default create_autograde setup test grade clean