import torch
import random
def flaw_generation(
    num_envs, bodydim=12, fixed_joint=[-1], flawed_rate=-1, device="cpu", upper_bound=1
):
    """
    num_envs: parallel envs
    fixed_joint: id of joint, -1 for randomization
    flawed_rate: degeneration rate, -1 for randomization

    Outputs: bodies, joints
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
    joint: (num_envs, num) OR a single int
    rate: In every step, w.p. rate to degenerate the joint worse.
    threshold: if the degenerate rate is lower than threshold, it will be set to a random value between 0.5 and 1.
    '''        
    num_envs = bodies.shape[0]
    t = torch.rand(num_envs)
    t = (t<rate) * torch.rand(num_envs)
    t = 1 - t
    t = t.to(bodies.device)
    if type(joint) == torch.Tensor:
        joint = joint.to(bodies.device)
        p = torch.gather(bodies, 1, joint) * t
        bodies = torch.scatter(bodies, 1, joint, p)
        if threshold > 0: 
            rands = torch.rand_like(bodies)/2 + 0.5
            rands = torch.clamp(rands, min=0, max=upper_bound)
            bodies = torch.where(bodies>threshold, bodies, rands)
    else:
        bodies[:, joint] *= t
        if threshold > 0:
            rands = torch.rand_like(bodies)/2 + 0.5
            rands = torch.clamp(rands, min=0, max=upper_bound)
            bodies = torch.where(bodies>threshold, bodies, rands)

    return bodies
