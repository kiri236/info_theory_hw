import torch
from typing import Union,Sequence,Any
def Calculate_Matrix_Entropy(Mat:Union[torch.Tensor,Sequence[Any]]):
    if not isinstance(Mat,torch.Tensor):
        Mat = torch.tensor(Mat)
    with torch.no_grad():
        eig_val = torch.svd(Mat / torch.trace(Mat))[1]
        entropy = -(eig_val*torch.log(eig_val)).nansum().item()
    return entropy
