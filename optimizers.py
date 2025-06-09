from torch import optim
import torch
import math
from torch.optim import Optimizer

class AdamWWithDenomClip(optim.AdamW):
    """
    一个自定义的AdamW优化器，增加了对更新分母进行裁剪的功能。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, denom_clip_value=None, denom_clip_type='max'):
        if denom_clip_value is not None and denom_clip_type not in ['max', 'min']:
            raise ValueError(f"denom_clip_type must be 'max' or 'min', but got {denom_clip_type}")
        
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.denom_clip_value = denom_clip_value
        self.denom_clip_type = denom_clip_type
        # 用于记录裁剪率的实例变量
        self.step_clip_rates = []

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 清空上一步的裁剪率记录
        self.step_clip_rates.clear()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])
            
            # --- 我们在这里介入，手动执行step的部分逻辑以插入裁剪 ---
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # --- 核心修改：计算并裁剪分母 ---
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if self.denom_clip_value is not None:
                    original_denom = denom.clone()
                    if self.denom_clip_type == 'max':
                        torch.clamp(denom, max=self.denom_clip_value, out=denom)
                        clipped_mask = original_denom > self.denom_clip_value
                    else: # min
                        torch.clamp(denom, min=self.denom_clip_value, out=denom)
                        clipped_mask = original_denom < self.denom_clip_value
                    
                    rate = torch.mean(clipped_mask.float()).item()
                    self.step_clip_rates.append(rate)

                # --- 结束修改 ---

                # 计算步长
                step_size = group['lr'] / bias_correction1
                
                # 应用权重衰减并更新参数
                if group['weight_decay'] != 0:
                    param.add_(param, alpha=-group['lr'] * group['weight_decay'])
                
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss