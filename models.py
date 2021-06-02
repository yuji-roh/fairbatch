import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def weights_init_normal(m):
    """Initializes the weight and bias of the model.

    Args:
        m: A torch model to initialize.

    Returns:
        None.
    """
    
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)


class LogisticRegression(nn.Module):
    """Logistic Regression (classifier).

    Attributes:
        model: A model consisting of torch components.
    """
    
    def __init__(self, n_in, n_out):
        """Initializes classifier with torch components."""
        
        super(LogisticRegression, self).__init__()

    
        def block(in_feat, out_feat, normalize=True):
            """Defines a block with torch components.
            
                Args:
                    in_feat: An integer value for the size of the input feature.
                    out_feat: An integer value for the size of the output feature.
                    normalize: A boolean indicating whether normalization is needed.
                    
                Returns:
                    The stacked layer.
            """
            
            layers = [nn.Linear(in_feat, out_feat)]
            
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
#             *block(3, 32, normalize=False),
#             nn.Linear(32, 1),
#             nn.Tanh()
            nn.Linear(n_in,n_out)
        )

    def forward(self, input_data):
        """Defines a forward operation of the model.
        
        Args: 
            input_data: The input data.
            
        Returns:
            The predicted label (y_hat) for the given input data.
        """
        
        output = self.model(input_data)
        return output
    
    
    
def test_model(model_, X, y, s1):
    """Tests the performance of a model.

    Args:
        model_: A model to test.
        X: Input features of test data.
        y: True label (1-D) of test data.
        s1: Sensitive attribute (1-D) of test data.

    Returns:
        The test accuracy and the fairness metrics of the model.
    """
    
    model_.eval()
    
    y_hat = model_(X).squeeze()
    prediction = (y_hat > 0.0).int().squeeze()
    y = (y > 0.0).int()

    z_0_mask = (s1 == 0.0)
    z_1_mask = (s1 == 1.0)
    z_0 = int(torch.sum(z_0_mask))
    z_1 = int(torch.sum(z_1_mask))
    
    y_0_mask = (y == 0.0)
    y_1_mask = (y == 1.0)
    y_0 = int(torch.sum(y_0_mask))
    y_1 = int(torch.sum(y_1_mask))
    
    Pr_y_hat_1 = float(torch.sum((prediction == 1))) / (z_0 + z_1)
    
    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1
        
    
    y_1_z_0_mask = (y == 1.0) & (s1 == 0.0)
    y_1_z_1_mask = (y == 1.0) & (s1 == 1.0)
    y_1_z_0 = int(torch.sum(y_1_z_0_mask))
    y_1_z_1 = int(torch.sum(y_1_z_1_mask))
    
    Pr_y_hat_1_y_0 = float(torch.sum((prediction == 1)[y_0_mask])) / y_0
    Pr_y_hat_1_y_1 = float(torch.sum((prediction == 1)[y_1_mask])) / y_1
    
    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1
    
    y_0_z_0_mask = (y == 0.0) & (s1 == 0.0)
    y_0_z_1_mask = (y == 0.0) & (s1 == 1.0)
    y_0_z_0 = int(torch.sum(y_0_z_0_mask))
    y_0_z_1 = int(torch.sum(y_0_z_1_mask))

    Pr_y_hat_1_y_0_z_0 = float(torch.sum((prediction == 1)[y_0_z_0_mask])) / y_0_z_0
    Pr_y_hat_1_y_0_z_1 = float(torch.sum((prediction == 1)[y_0_z_1_mask])) / y_0_z_1
    
    recall = Pr_y_hat_1_y_1
    precision = float(torch.sum((prediction == 1)[y_1_mask])) / (int(torch.sum(prediction == 1)) +0.00001)
    
    y_hat_neq_y = float(torch.sum((prediction == y.int())))

    test_acc = torch.sum(prediction == y.int()).float() / len(y)
    test_f1 = 2 * recall * precision / (recall+precision+0.00001)
    
    min_dp = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1) + 0.00001
    max_dp = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1) + 0.00001
    min_eo_0 = min(Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1) + 0.00001
    max_eo_0 = max(Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1) + 0.00001
    min_eo_1 = min(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1) + 0.00001
    max_eo_1 = max(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1) + 0.00001
    
    DP = max(abs(Pr_y_hat_1_z_0 - Pr_y_hat_1), abs(Pr_y_hat_1_z_1 - Pr_y_hat_1))
    
    EO_Y_0 = max(abs(Pr_y_hat_1_y_0_z_0 - Pr_y_hat_1_y_0), abs(Pr_y_hat_1_y_0_z_1 - Pr_y_hat_1_y_0))
    EO_Y_1 = max(abs(Pr_y_hat_1_y_1_z_0 - Pr_y_hat_1_y_1), abs(Pr_y_hat_1_y_1_z_1 - Pr_y_hat_1_y_1))

    
    return {'Acc': test_acc.item(), 'DP_diff': DP, 'EO_Y0_diff': EO_Y_0, 'EO_Y1_diff': EO_Y_1, 'EqOdds_diff': max(EO_Y_0, EO_Y_1)}

#     return {'Acc':test_acc.item(), 'F1':test_f1, 'DP_ratio':min_dp/max_dp, 'EO_Y0_ratio':min_eo_0/max_eo_0, 'EO_Y1_ratio':min_eo_1/max_eo_1, 
#             'DP_diff':DP, 'EO_Y0_diff':EO_Y_0, 'EO_Y1_diff':EO_Y_1, 'EqOdds_diff':max(EO_Y_0, EO_Y_1), 'Prob':[Pr_y_hat_1_z_0, Pr_y_hat_1_z_1, Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1, Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1]}



