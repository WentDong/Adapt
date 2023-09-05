import torch
import random
def flaw_generation(
    num_envs, bodydim=12, fixed_joint=[-1], flawed_rate=-1, device="cpu", upper_bound=1
):
    """
    num_envs: 环境数
    fixed_joint: 指定损坏的关节为fixed_joint(LIST) [0,11]，若不固定为-1
    flawed_rate: 损坏程度为flawed_rate, 若随机坏损为-1
    t(num_envs * len(fixed_joint)): 坏损的关节
    """
    if bodydim == 0:
        return None, None
    if type(fixed_joint) is not list:
        fixed_joint = [fixed_joint]
    t = torch.randint(0, bodydim, (num_envs, 1))
    if -1 not in fixed_joint:
        t = torch.ones((num_envs, len(fixed_joint)), dtype=int) * torch.tensor(
            fixed_joint
        )
    bodies = torch.ones(num_envs, bodydim).to(device)
    for i in range(num_envs):
        for joint in [t[i]]:
            bodies[i, joint] = (
                random.random() * upper_bound if flawed_rate == -1 else flawed_rate
            )
    return bodies, t

def step_body(bodies, joint, rate = 0.004, threshold = 0, upper_bound=1): #each joint has a flaw rate to be partial of itself.
    '''
    joint: (num_envs, num) OR a single int, 每个环境对应的1个坏损关节 
        #TODO: joint will become (num_envs, num), num means the number of flawed joints.
    rate: 每个step, 有rate的概率使得关节扭矩往下掉, 剩余扭矩比例随机
    threshold, 在剩余扭矩低于threshold时, 重置到随机的一个扭矩。
    '''        
    num_envs = bodies.shape[0]
    t = torch.rand(num_envs)
    t = (t<rate) * torch.rand(num_envs)
    t = 1 - t
    t = t.to(bodies.device)
    if type(joint) == torch.Tensor:
        joint = joint.to(bodies.device)
        # print(bodies.shape, joint.shape, t.shape)
        p = torch.gather(bodies, 1, joint) * t
        bodies = torch.scatter(bodies, 1, joint, p)
        if threshold > 0: 
            # tmp = torch.gather(bodies, 1, joint)
            # t = (tmp < threshold) * torch.rand(num_envs, device=bodies.device)
            # t = t.to(bodies.device)
            # t = 1 / (1 - t)
            # bodies = torch.scatter(bodies, 1, joint, t * tmp)
            rands = torch.rand_like(bodies)/2 + 0.5
            rands = torch.clamp(rands, min=0, max=upper_bound)
            bodies = torch.where(bodies>threshold, bodies, rands)
            # bodies = torch.clamp(bodies, min=0, max=upper_bound)
    else:
        bodies[:, joint] *= t
        if threshold > 0:  # Here we assume that joint must be a single int
            # t = (bodies[:, joint] < threshold) * torch.rand(num_envs, device=bodies.device) * (torch.rand(num_envs, device= bodies.device) < rate_reset)
            # t = t.to(bodies.device)
            # t = 1 / (1 - t)
            # bodies[:, joint] *= t
            rands = torch.rand_like(bodies)/2 + 0.5
            rands = torch.clamp(rands, min=0, max=upper_bound)
            bodies = torch.where(bodies>threshold, bodies, rands)
            # bodies = torch.clamp(bodies, min=0, max=upper_bound)

    return bodies
