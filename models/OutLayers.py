import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F



class OutOpenEnded(nn.Module):
    def __init__(self, embed_dim=512, num_answers=1000, drorate=0.1):
        super(OutOpenEnded, self).__init__()
        self.activ=nn.GELU()

        self.classifier = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, num_answers))
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, Out):
        out = self.classifier(Out)
        return out



class OutCount(nn.Module):
    def __init__(self, embed_dim=512, drorate=0.1):
        super(OutCount, self).__init__()
        self.activ=nn.GELU()

        self.regression = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, Out):
        out = self.regression(Out)
        return out



class OutMultiChoices(nn.Module):
    def __init__(self, embed_dim=512, drorate=0.1):
        super(OutMultiChoices, self).__init__()
        self.activ=nn.GELU()

        self.classifier = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim*2, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, Out, Out_an_expand):
        out = self.classifier(torch.cat([Out, Out_an_expand], -1))
        return out