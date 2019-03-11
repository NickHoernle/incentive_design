import pandas as pd
import numpy as np
import scipy.stats as stats
from  scipy.special import logsumexp
from multiprocessing import Pool

# how many available processors can we use for this computation.
numprocs = 3

def sample_assignments(user_ix, action_trajectory, params):
    actions = np.zeros(params['vocab_size'])
    num_profiles = params['num_profiles']
    phi = params['phi']
    theta = params['theta']
    assignment_choices = params['assignment_choices']

    for action_ix, action_count in action_trajectory:
            actions[action_ix] = action_count

    total_action_count = np.sum(actions)
    likelihood = np.array([stats.multinomial(n=total_action_count, p=np.exp(phi[k])).logpmf(actions) for k in range(num_profiles)], dtype=np.float32)
    likelihood += theta

    # normalize
    likelihood -= logsumexp(likelihood)

    assignment = np.random.choice(assignment_choices, p=np.exp(likelihood))

    return (user_ix, assignment)

def run_sampler(user_actions, num_iter=50, num_profiles=10, alpha=[], beta=[], vocab_size=0):

    assignment_choices = np.arange(num_profiles)
    if vocab_size == 0:
        if len(beta) == 0:
            print('One of beta or vocab_size must be set')
        vocab_size = len(beta)

    if len(beta) == 0:
        # default prior
        beta = np.ones(vocab_size)

    if len(alpha) == 0:
        # default prior
        alpha = np.ones(num_profiles)

    # initialise theta - prior over what category you belong in
    theta = np.log(np.random.dirichlet(alpha))

    # initialise phi - prior over a categories distribution over actions
    phi = np.log(np.random.dirichlet(alpha=beta, size=num_profiles))

    print(theta.shape)
    print(phi.shape)

    params = {
        'vocab_size': vocab_size,
        'alpha': alpha,
        'beta': beta,
        'phi': phi,
        'theta': theta,
        'assignment_choices': assignment_choices,
        'num_profiles': num_profiles
    }

    # randomly assign a user to a profile class
    assignments = np.random.choice(assignment_choices, p=theta, size=len(user_actions))

    for epoch in range(num_iter):

        # update the user assignment
        # let's speed up this computation
        results = []
        pool = Pool(numprocs)
        for u, user in enumerate(user_actions):
            results.append(pool.apply_async(sample_assignments, args=(u, user, params, )))

        for result in results:
            res_ = result.get()
            assignments[res_[0]] = res_[1]

        pool.close()
        pool.join()
        # update the action distribution for each of the topics
        for k in range(num_profiles):
            users_in_this_profile = [user_actions[j] for j in np.where(assignments == k)[0]]

            # prior
            actions = np.zeros(vocab_size) + beta

            for user in users_in_this_profile:
                for action_ix, action_count in user:
                    actions[action_ix] += action_count

            # sample from the complete conditional
            phi[k] = np.log(np.random.dirichlet(alpha=actions))

        params['phi'] = phi

        # update theta
        theta_ = np.array([np.sum(assignments==k) for k in range(num_profiles)])
        theta = np.log(np.random.dirichlet(alpha=theta_+alpha))
        params['theta'] = theta

        print(epoch, theta)

    # note for now we are just returning one point estimate from the posterior
    # distribution. This should rather return a mean or something better than
    # one sample.
    return {
        'theta': theta,
        'phi': phi,
        'assignments': assignments
    }
