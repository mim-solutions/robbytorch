import torch
from tqdm import tqdm

from . import utils, attack_steps



STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep,
    'fourier': attack_steps.FourierStep,
    'random_smooth': attack_steps.RandomStep
}


def PGD(model, dataitem, forward, step_size, Nsteps, loss_key='loss',
        constraint='2', eps=None, minimize=False, random_start=False, use_tqdm=False):
    '''
    Compute adversarial examples for given model.
    Args:
        model: model
        dataitem: batched dataitem structure
        forward: Callable taking model and dataitem and returning {"loss": loss, ...}
        step_size (float): optimization step size
        Nsteps (int): number of optimization steps
        eps (float): radius of L2 ball
        minimize (bool): True if we want to minimize the loss, else False
    Returns:
        x: batch of adversarial examples for input images
    '''
    prev_training = bool(model.training)
    model.eval()

    x = dataitem["data"].detach()
    original_device = utils.get_device(x)
    step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
    step = step_class(eps=eps, orig_input=x, step_size=step_size)

    if random_start:
        x = step.random_perturb(x)
    
    sign = -1 if minimize else 1

    it = iter(range(Nsteps))
    if use_tqdm:
        it = tqdm(iter(range(Nsteps)), total=Nsteps) 

    for i in it:    
        x = x.clone().detach().requires_grad_(True)
        loss = forward(model, {**dataitem, "data": x}, "eval")[loss_key]
        (grad,) = torch.autograd.grad(loss, [x])

        if use_tqdm:
            it.set_description(f'Loss: {loss.item()}')
        
        with torch.no_grad():            
            x = step.step(x, sign * grad)
            x = step.project(x)
    
    if prev_training:
        model.train()

    return x.detach().to(original_device)