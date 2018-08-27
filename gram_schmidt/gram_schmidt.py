#!/usr/bin/env python

import torch
import torch.nn as nn

class GramSmidth(nn.Module):
    def __init__(self):
        super(GramSmidth, self).__init__()

    def forward(self, x):
        o = x.new()
        for i in range(len(x)):
            vx = x[i,0]
            vy = x[i,1]
            vz = x[i,2]
            ux = torch.zeros_like(vx)
            uy = torch.zeros_like(vx)
            uz = torch.zeros_like(vx)
            ex = torch.zeros_like(vx)
            ey = torch.zeros_like(vx)
            ez = torch.zeros_like(vx)


            ux = vx
            uy = vy - ((torch.dot(ux,vy)*ux)/torch.dot(ux,ux))
            uz = vz - ((torch.dot(ux,vz)*ux)/torch.dot(ux,ux)) - ((torch.dot(uy,vz)*uy)/torch.dot(uy,uy))

            ex = ux / torch.norm(ux)
            ey = uy / torch.norm(uy)
            ez = uz / torch.norm(uz)

            ee = torch.cat((ex, ey, ez))

            o = torch.cat((o,ee.view(1,3,3)))

        return o




#rnn[0] = rn[0]
#rnn[1] = rn[1] - (rn[0].dot(rn[1])*rn[0])/rn[0].dot(rn[0])
#rnn[2] = rn[2] - (rn[0].dot(rn[2])*rn[0])/rn[0].dot(rn[0]) - (rnn[1].dot(rn[2])*rnn[1])/rnn[1].dot(rnn[1])
#
#rnn[0] /= np.linalg.norm(rnn[0])
#rnn[1] /= np.linalg.norm(rnn[1])
#rnn[2] /= np.linalg.norm(rnn[2])
#
#
#x[1] = x[1] - ((torch.dot(x[0],x[1])*x[0])/torch.dot(x[0],x[0]))
#x[2] = x[2] - ((torch.dot(x[0],x[2])*x[0])/torch.dot(x[0],x[0])) - ((torch.dot(x[1],x[2])*x[1])/torch.dot(x[1],x[1]))
#
#x[0] = x[0] / torch.norm(x[0])
#x[1] = x[1] / torch.norm(x[1])
#x[2] = x[2] / torch.norm(x[2])
#
#
#x = ii*pp
#
#
#        b, m, n = x.size()
#        Q = torch.zeros_like(x, requires_grad=False)
#        R = torch.zeros_like(x, requires_grad=False)
#        V = x.clone()
#
#        for i in range(n):
#            R[:,i,i] = V[:,:,i].detach().norm(dim=1)
#            Q[:,:,i] = V[:,:,i].clone() / R[:,i,i].unsqueeze(0).transpose(0,1).repeat(1,n)
#
#            for j in range(i, n):
#                R[:, i,j] = torch.bmm(Q[:,:,i].detach().view(b, 1, m), V[:,:,j].detach().view(b, m, 1)).squeeze()
#                V[:,:,j] = V[:,:,j].clone() - R[:,i,j].unsqueeze(0).transpose(0,1).repeat(1,3)*Q[:,:,i].detach()
