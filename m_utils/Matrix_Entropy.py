import torch
from typing import Union,Sequence,Any,List
import math
def dimension(Mat:Union[torch.Tensor,Sequence[Any]])->int:
    if hasattr(Mat,"shape"):
        return len(Mat.shape)
    dim = 1
    while isinstance(Mat[0],List):
        dim += 1
        Mat = Mat[0]
    return dim
def Calculate_Matrix_Entropy_raw(Mat:Union[torch.Tensor,Sequence[Any]])->Union[float,List]:
    if dimension(Mat) >= 3:
        entropy_list = []
        for mat in Mat:
            ans = Calculate_Matrix_Entropy_raw(mat)
            entropy_list.append(ans)
        return entropy_list
    if not isinstance(Mat,torch.Tensor):
        Mat = torch.tensor(Mat)
    with torch.no_grad():
        eig_val = torch.svd(Mat / torch.trace(Mat))[1]
        entropy = -(eig_val*torch.log(eig_val)).nansum().item()
    return entropy
def Calculate_Matrix_Entropy(Mat:Union[torch.Tensor,Sequence[Any]])->torch.Tensor:
    return torch.tensor(Calculate_Matrix_Entropy_raw(Mat))

print(Calculate_Matrix_Entropy(([[[[1,2,3],[2,3,4]],[[5,6,5],[34,5,3]]],[[[1,4,2],[2,4,3]],[[5,2,6],[34,2,3]]]])))