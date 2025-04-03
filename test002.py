import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
torch.set_printoptions(precision = 8)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class InteguralCheck:
    def __init__(self):
        self.dT = []
        self.x__ = []
        self.t__ = []
        self.xnp2 = []
        
    def init_values(self):
        self.x__ = torch.tensor([0.0])
        self.t__ = torch.tensor([0.0])
        self.dT  = torch.tensor([0.001])
        
    def RungeKutta4th(self,fnc_Acc, dt, tuple_other_args):
        mat_a = torch.tensor([[    0.0,     0.0, 0.0, 0.0],
                              [1.0/2.0,     0.0, 0.0, 0.0],
                              [    0.0, 1.0/2.0, 0.0, 0.0],
                              [    0.0,     0.0, 1.0, 1.0]])
        mat_b = torch.tensor([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
        mat_c = mat_a.sum(dim=1)
        return mat_a, mat_b, mat_c

    def BogackiShampine(self):
        mat_a = torch.tensor([[     0.0,    0.0,    0.0,    0.0],
                              [1.0/ 2.0,    0.0,    0.0,    0.0],
                              [     0.0,3.0/4.0,    0.0,    0.0],
                              [2.0/ 9.0,3.0/9.0,4.0/9.0,    0.0]])
        mat_b = torch.tensor([[2.0/ 9.0,3.0/9.0,4.0/9.0,    0.0],
                              [7.0/24.0,1.0/4.0,1.0/3.0,1.0/8.0]])
        mat_c = mat_a.sum(dim=1)
        return mat_a, mat_b, mat_c
    def RungeKuttaFehlberg(self):
        mat_a = torch.tensor([[            0.0,             0.0,             0.0,            0.0,            0.0, 0.0],
                              [    1.0/    4.0,             0.0,             0.0,            0.0,            0.0, 0.0],
                              [    3.0/   32.0,     9.0/   32.0,             0.0,            0.0,            0.0, 0.0],
                              [ 1932.0/ 2197.0, -7200.0/ 2197.0,  7296.0/ 2197.0,            0.0,            0.0, 0.0],
                              [ 8341.0/ 4104.0,-32832.0/ 4104.0, 29440.0/ 4104.0, -845.0/ 4104.0,            0.0, 0.0],
                              [-6080.0/20520.0,  4140.0/20520.0,-28352.0/20520.0, 9295.0/20520.0,-5643.0/20520.0, 0.0]])
        mat_b = torch.tensor([[   25.0/  216.0,             0.0,  1408.0/ 2565.0, 2197.0/ 4104.0,   -1.0/    5.0, 0.0],
                              [   16.0/  135.0,             0.0,  6656.0/12825.0,28561.0/56430.0,   -9.0/   50.0, 2.0/55.0]])
        mat_c = mat_a.sum(dim=1)
        return mat_a, mat_b, mat_c
        
    def DormandPrince(self):
        mat_a = torch.tensor([[            0.0,            0.0,             0.0,         0.0,              0.0,         0.0, 0.0],
                              [     1.0/   5.0,            0.0,             0.0,         0.0,              0.0,         0.0, 0.0],
                              [     3.0/  40.0,     9.0/  40.0,             0.0,         0.0,              0.0,         0.0, 0.0],
                              [    44.0/  45.0,   -56.0/  15.0,    32.0/    9.0,         0.0,              0.0,         0.0, 0.0],
                              [ 19372.0/6561.0,-25360.0/2187.0, 64448.0/ 6561.0,-212.0/729.0,              0.0,         0.0, 0.0],
                              [  9017.0/3168.0,  -355.0/  33.0, 46732.0/ 5247.0,  49.0/176.0, -5103.0/ 18656.0,         0.0, 0.0],
                              [    35.0/ 384.0,            0.0,   500.0/ 1113.0, 125.0/192.0, -2187.0/  6784.0, 11.0/  84.0, 0.0]])
        mat_b = torch.tensor([ [   35.0/ 384.0,            0.0,   500.0/ 1113.0, 125.0/192.0, -2187.0/  6784.0, 11.0/  84.0, 0.0],
                              [ 5179.0/57600.0,            0.0,  7571.0/16695.0, 393.0/640.0,-92097.0/339200.0,187.0/2100.0, 1.0/40.0]])
        mat_c = mat_a.sum(dim=1)
        return mat_a, mat_b, mat_c
    
    def RungeKuttaVerner(self):
        mat_a = torch.tensor([[            0.0,            0.0,             0.0,             0.0,            0.0,        0.0,           0.0, 0.0],
                              [     1.0/  18.0,            0.0,             0.0,             0.0,            0.0,        0.0,           0.0, 0.0],
                              [    -1.0/  12.0,     3.0/  12.0,             0.0,             0.0,            0.0,        0.0,           0.0, 0.0],
                              [    -2.0/  81.0,    12.0/  81.0,      8.0/  81.0,             0.0,            0.0,        0.0,           0.0, 0.0],
                              [    40.0/  33.0,   -12.0/  33.0,   -168.0/  33.0,    162.0/  33.0,            0.0,        0.0,           0.0, 0.0],
                              [ -8856.0/1752.0,  1728.0/1752.0, -43040.0/1752.0,  36855.0/1752.0, -2695.0/1752.0,        0.0,           0.0, 0.0],
                              [ -8716.0/ 891.0,  1968.0/ 891.0,  39520.0/ 891.0, -33696.0/ 891.0,  1716.0/ 891.0,        0.0,           0.0, 0.0],
                              [117585.0/9984.0,-22464.0/9984.0,-540032.0/9984.0, 466830.0/9984.0,-14014.0/9984.0,        0.0, 2079.0/9984.0, 0.0]])
        mat_b = torch.tensor([[     3.0/  80.0,            0.0,      4.0/  25.0,    243.0/1120.0,    77.0/ 160.0, 73.0/700.0,           0.0, 0.0],
                              [    57.0/ 640.0,            0.0,    -16.0/  65.0,   1377.0/2240.0,   121.0/ 320.0,        0.0,  891.0/8320.0, 2.0/35.0]])
                
        mat_c = mat_a.sum(dim=1)
        return mat_a, mat_b, mat_c
            
    def integer_selct(self,fnc_Acc, dt, tuple_other_args, type_of_inter):
        
        if type_of_inter == 0:
            mat_a, mat_b, mat_c = self.RungeKutta4th(fnc_Acc,dt,tuple_other_args)

            dt_ = dt
            vec_k = torch.zeros((self.x__.shape[0],mat_a.shape[1]))
            for ii in range(mat_a.shape[0]):
                vec_k[:,ii] = fnc_Acc(self.t__+mat_c[ii]*dt_, self.x__+dt_*torch.matmul(vec_k, mat_a[ii,:]), *tuple_other_args)
            self.xnp2 = self.x__ + dt_ * (torch.matmul(vec_k, mat_b))
        else:
            if type_of_inter == 1:
                error_odr = 3.0
                mat_a, mat_b, mat_c = self.BogackiShampine()
            elif type_of_inter == 2:
                error_odr = 5.0
                mat_a, mat_b, mat_c = self.RungeKuttaFehlberg()
            elif type_of_inter == 3:
                error_odr = 5.0
                mat_a, mat_b, mat_c = self.DormandPrince()
            elif type_of_inter == 4:
                error_odr = 7.0
                mat_a, mat_b, mat_c = self.RungeKuttaVerner()
            
            dt_ = dt.clone().detach() if isinstance(dt, torch.Tensor) else torch.tensor(dt)
            while True:
                vec_k = torch.zeros((self.x__.shape[0],mat_a.shape[1]))
                for ii in range(mat_a.shape[0]):
                    vec_k[:,ii] = fnc_Acc(self.t__+mat_c[ii]*dt_, self.x__+dt_*torch.matmul(vec_k, mat_a[ii,:]), *tuple_other_args)
                xnp1 = self.x__.clone().detach() + dt_ * (torch.matmul(vec_k, mat_b[0,:]))
                xnp2 = self.x__.clone().detach() + dt_ * (torch.matmul(vec_k, mat_b[1,:]))
                
                dt_, isbreak = self.fix_dt(xnp1, xnp2, dt_, error_odr)
                if isbreak:
                    dt_ = dt_.item()
                    break
                
        self.re_dt = dt_
            
        return self.xnp2

    def fix_dt(self, xnp1, xnp2, dt_, error_odr):
        alpha = 0.9
        error_tol = 0.0001
        
        isbreak = False
        T_error = torch.abs(xnp1-xnp2)
        if T_error.max()/error_tol < 1.0e-3:
            dt_ = torch.min(alpha * dt_ * (error_tol / (2.0 * T_error.max() + 1.0e-8))**(1.0 / error_odr), dt_.clone().detach())         
        elif T_error.max() < error_tol:
            isbreak = True
        else:
            dt_ = alpha*dt_*(error_tol/(2.0*T_error.max()+1.0e-8))**(1.0/error_odr)
            
        return dt_, isbreak


#%%
if __name__ == '__main__':
    print('test start')
    
    def fnc_function(t, x):
        return torch.sin(t)*torch.exp(t)
    
    def fnc_ans_function(t, x):
        return  0.5*(torch.sin(t)*torch.exp(t)-torch.cos(t)*torch.exp(t)) + 0.5
    
    
    for ii in range(1,4):
        clsInt = InteguralCheck()
        clsInt.dT = torch.tensor(0.0001)
        clsInt.init_values()
        
        # 数値解を求める
        t_values = []
        x_values = []
        x_true   = []
        dt_ = clsInt.dT
        while clsInt.t__ < 1.0:
            x_true.append(fnc_ans_function(clsInt.t__.clone(),clsInt.x__.clone()))
            
            clsInt.integer_selct(fnc_function,dt_,(),ii)
            clsInt.x__  = clsInt.xnp2.clone()
            clsInt.t__ += clsInt.re_dt
            
            tt = clsInt.t__.item()
            t_values.append(tt)
            x_values.append(clsInt.x__.clone())
            
            
            progress = tt/1.0
            block = int(40 * progress)
            bar   = "#" * block + "-"*(40-block)
            sys.stdout.write(f"\r[{bar}] {tt}/{1.0} ({progress*100:.2f}%)")
            sys.stdout.flush()

        x_values = torch.cat(x_values,dim=0).to('cpu').numpy()
        x_true   = torch.cat(x_true,dim=0).numpy()
        t_values = np.array(t_values)

        # 誤差を計算する
        errors = np.abs(x_values-x_true)

        # 結果を表示
        print("Numerical Solution:", x_values[-1])
        print("Analytical Solution:", x_true[-1])
        print("Error:", errors[-1])
        
        # プロット
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # 数値解と解析解のプロット
        axs[0].plot(t_values, x_values, label='Numerical Solution')
        axs[0].plot(t_values, x_true, label='Analytical Solution', linestyle='dashed')
        axs[0].set_title('Numerical vs Analytical Solution')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Value')
        axs[0].legend()

        # 誤差のプロット
        axs[1].plot(t_values, errors, label='Error')
        axs[1].set_title('Error over Time')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Error')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
