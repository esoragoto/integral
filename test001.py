
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
torch.set_printoptions(precision = 8)
torch.set_default_dtype(torch.float64)

def ensure_tensor_column_vector(xn):
    if not isinstance(xn, torch.Tensor):
        xn = torch.tensor(xn, dtype=torch.float64)
    return xn
def fnc_AddaptiveRungeKutta(dt, tn, xn, fnc_F):
    # https://park.itc.u-tokyo.ac.jp/YNagai/oldstyle/note_131119_RungeKutta.pdf
    def fix_dt(xnp1, xnp2,dt_, error_odr):
        alpha     = 0.99
        error_tol = 1e-8

        isbreak = False
        T_error = abs(xnp1-xnp2)
        
        if torch.all((T_error / error_tol) < 1e-8):
            dt_ = min(min(alpha * dt_ * (error_tol / 2 / (T_error + 1e-10)) ** (1 / error_odr)), dt_)
        if torch.sum(T_error > error_tol) == 0:
            isbreak = True
        else:
            dt_ = min(alpha * dt_ * (error_tol / 2 / (T_error + 1e-10)) ** (1 / error_odr))


        return dt_, isbreak
    
    error_odr = 7
    mat_a = np.array([np.array([        0.0,      0.0,        0.0,           0.0,        0.0,         0.0,          0.0,      0.0])/1.0,\
                      np.array([        1.0,      0.0,        0.0,           0.0,        0.0,         0.0,          0.0,      0.0])/18.0,\
                      np.array([       -1.0,      3.0,        0.0,           0.0,        0.0,         0.0,          0.0,      0.0])/12.0,\
                      np.array([       -2.0,     12.0,        8.0,           0.0,        0.0,         0.0,          0.0,      0.0])/81.0,\
                      np.array([       40.0,    -12.0,     -168.0,         162.0,        0.0,         0.0,          0.0,      0.0])/33.0,\
                      np.array([    -8856.0,   1728.0,    43040.0,      -36855.0,     2695.0,         0.0,          0.0,      0.0])/1752.0,\
                      np.array([    -8716.0,   1968.0,    39520.0,      -33696.0,     1716.0,         0.0,          0.0,      0.0])/891.0,\
                      np.array([   117585.0, -22464.0,  -540032.0,      466830.0,   -14014.0,         0.0,       2079.0,      0.0])/9984.0],\
                      dtype=np.float64)
    mat_b = np.array([np.array([   3.0/80.0,      0.0,   4.0/25.0,  243.0/1120.0,  77.0/160.0, 73.0/700.0,          0.0,      0.0]),\
                      np.array([ 57.0/640.0,      0.0, -16.0/65.0, 1377.0/2240.0, 121.0/320.0,        0.0, 891.0/8320.0, 2.0/35.0])],\
                      dtype=np.float64)
    mat_c = np.sum(mat_a, axis=1)        
    
    # numpy to tensor
    mat_a = torch.tensor(mat_a, dtype=torch.float64)
    mat_b = torch.tensor(mat_b, dtype=torch.float64)
    mat_c = torch.tensor(mat_c, dtype=torch.float64)
    vec_k = torch.zeros([xn.shape[0], mat_a.shape[1]], dtype=torch.float64)
    
    xn = ensure_tensor_column_vector(xn)
    dt_ = dt
    while(True):
        # calc. ki = f(tn + ci*dt, xn + sum(aj*ki*dt))
        
        mat_tn_dt = tn + mat_c*dt_
        for ii in range(0, np.shape(mat_a)[1]):
            mat_xn_dt = xn + torch.matmul(vec_k, mat_a[ii,:])*dt_    
            vec_k[:,ii] = fnc_F(mat_tn_dt[ii], mat_xn_dt)
            
        xnp1 = xn + dt_ * (torch.matmul(vec_k, mat_b[0,:]))
        xnp2 = xn + dt_ * (torch.matmul(vec_k, mat_b[1,:]))
        [dt_, isbreak] = fix_dt(xnp1, xnp2, dt_, error_odr)

        if isbreak:
            break
    
    return xnp2.reshape((1, -1))[0], dt_


#%%
if __name__ == '__main__':
    print('test start')
    
    def fnc_function(t, x):
        return torch.sin(t)*torch.exp(t)
    
    def fnc_ans_function(t, x):
        return  0.5*(torch.sin(t)*torch.exp(t)-torch.cos(t)*torch.exp(t)) + 0.5
    
    # 初期条件
    dt = 0.0001
    tn = torch.tensor([0.0], dtype=torch.float64)
    xn = torch.tensor([0.0], dtype=torch.float64)
    
    # 数値解を求める
    t_values = []
    x_values = []
    x_true   = []
    dt_ = dt
    while tn < 1.0:
        x_true.append(fnc_ans_function(tn.clone(),xn.clone()))
        
        xn, dt_ = fnc_AddaptiveRungeKutta(dt_, tn, xn, fnc_function)
        tn += dt_
        t_values.append(tn.item())
        x_values.append(xn)
        
        
        progress = tn.item()/10
        block = int(40 * progress)
        bar   = "#" * block + "-"*(40-block)
        sys.stdout.write(f"\r[{bar}] {tn.item()}/{10} ({progress*100:.2f}%)")
        sys.stdout.flush()

    x_values = torch.cat(x_values,dim=0).numpy()
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
# %%
