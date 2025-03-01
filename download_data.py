import gdown
data_link = "1-0e9Gnl4nm6wbIuE_Yxl2wRZI8yGxHm6"
solution_link = "10W49bC2qoHjltNI_4PDqoArZzdgZHBfy"
url = lambda link: f"https://drive.google.com/uc?id={link}"
gdown.download(url(data_link), "hw4_data.tar.gz")
gdown.download(url(solution_link), "hw4p2_sol.json")


