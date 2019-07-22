def test_negative_binomial():
    d = NegativeBinomial(10.0, 0.5)
    print(d.event_shape)
    print(d.batch_shape)
    value = torch.tensor(np.arange(0, 100, dtype=np.float)).float()
    p = np.exp(d.log_prob(value).numpy())
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(p[:100])
    ax.set_xlabel('Count', fontsize=16)
    ax.set_ylabel('PMF', fontsize=16)
    print(f'Total probability: {np.sum(p)}')

    _ = plt.hist(d.sample([1_000_000]), density=True, bins=100, range=(0, 100))

def test_three_component_neg_binom_mixture():
    print('[TEST] Three component negative binomial mixture')
    dist_1 = NegativeBinomial(1.0, 0.5)
    dist_2 = NegativeBinomial(15.0, 0.1)
    dist_3 = NegativeBinomial(40.0, 0.1)
    log_weight_1 = torch.tensor([0.1]).log()
    log_weight_2 = torch.tensor([0.4]).log()
    log_weight_3 = torch.tensor([1.5]).log()
    d = MixtureDistribution(
        (log_weight_1, log_weight_2, log_weight_3),
        (dist_1, dist_2, dist_3),
        normalize_weights=True)

    values = torch.arange(0, 200).float()
    p = d.log_prob(values).exp().numpy()
    
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(values.numpy(), p)
    ax.set_xlabel('Count', fontsize=16)
    ax.set_ylabel('PMF', fontsize=16)
    print(f'Total probability: {np.sum(p)}')
    print()

def test_mixture_distribution_batch_shape():
    print('[TEST] Shapes in mixture distribution')
    dist_1 = NegativeBinomial(1.0, 0.5).expand([5, 2, 1])
    dist_2 = NegativeBinomial(15.0, 0.1).expand([2, 1])
    dist_3 = NegativeBinomial(40.0, 0.1)
    log_weight_1 = torch.tensor([[0.1], [0.2]]).log()
    log_weight_2 = torch.tensor([0.4]).log()
    log_weight_3 = torch.tensor([1.5]).log()
    d = MixtureDistribution(
        (log_weight_1, log_weight_2, log_weight_3),
        (dist_1, dist_2, dist_3),
        normalize_weights=True)

    values = torch.arange(0, 200).float().expand([10, 5, 2, 200])
    log_prob = d.log_prob(values)
    total_prob = torch.sum(log_prob.exp(), -1)
    assert torch.allclose(total_prob, torch.ones_like(total_prob))
    
    print(f'Component 1 (batch_shape: {dist_1.batch_shape}, event_shape: {dist_1.event_shape}')
    print(f'Component 2 (batch_shape: {dist_2.batch_shape}, event_shape: {dist_2.event_shape}')
    print(f'Component 3 (batch_shape: {dist_3.batch_shape}, event_shape: {dist_3.event_shape}')
    print(f'Weight 1 (shape: {log_weight_1.shape}')
    print(f'Weight 2 (shape: {log_weight_2.shape}')
    print(f'Weight 3 (shape: {log_weight_3.shape}')
    print()

    print(f'Component 1 in mixture distribution (batch_shape: {d.components[0].batch_shape}, event_shape: {d.components[0].event_shape}')
    print(f'Component 2 in mixture distribution (batch_shape: {d.components[1].batch_shape}, event_shape: {d.components[1].event_shape}')
    print(f'Component 3 in mixture distribution (batch_shape: {d.components[2].batch_shape}, event_shape: {d.components[2].event_shape}')
    print(f'Weight 1 in mixture distribution (shape: {d.log_weights[0].shape}')
    print(f'Weight 2 in mixture distribution (shape: {d.log_weights[1].shape}')
    print(f'Weight 3 in mixture distribution (shape: {d.log_weights[2].shape}')
    print()
    
    print(f'value shape: {values.shape}')
    print(f'log_prob shape: {log_prob.shape}')
    print()

    
def test_zero_inflated_negative_binomial():
    p_zero = 0.2
    mu = 15.0
    phi = 0.3
    zinb = ZeroInflatedNegativeBinomial(
        logit(torch.tensor(p_zero, dtype=torch.float64)),
        torch.tensor(mu, dtype=torch.float64),
        torch.tensor(phi, dtype=torch.float64))

    # total probability
    counts = torch.arange(0., 1000., dtype=torch.float64)
    log_prob = zinb.log_prob(counts)
    total_prob = log_prob.exp().sum()
    print(f'Total probability: {total_prob:.3f}')

    # sample
    zinb_samples = zinb.sample(sample_shape=[1_000_000])

    ax = plt.gca()
    ax.plot(0.5 + counts.numpy()[:50], log_prob.exp().numpy()[:50])
    _ = ax.hist(zinb_samples.numpy(), density=True, range=(0, 50), bins=50)
    ax.set_xlabel('counts')
    ax.set_ylabel('probability')

    # empirical mean and variance vs. analytical
    sample_mean, sample_var = torch.mean(zinb_samples), torch.std(zinb_samples).pow(2)
    true_mean, true_var = zinb.mean, zinb.variance
    assert torch.isclose(sample_mean, true_mean, atol=1.)
    assert torch.isclose(sample_var, true_var, atol=1.)
    
    
def test_get_confidence_interval():
    lower_cdf = 0.1
    upper_cdf = 0.9

    test_cdf_list = []
    expected_lo_idx_list = []
    expected_hi_idx_list = []

    test_cdf_list = [
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # trivial
        [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # right on lower_cdf and upper
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # all above lower_cdf
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # not properly normalized
        [0.09, 0.099999, 0.100001, 0.2, 0.3, 0.5, 1.0],  # small differences around lower_cdf
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.899999, 0.900001, 1.0],  # small differences around upper_cdf
        [0.95, 0.99, 1.0]  # all above upper_cdf
    ]  

    expected_lo_idx_list = [
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([0])
    ]

    expected_hi_idx_list = [
        torch.tensor([10]),
        torch.tensor([10]),
        torch.tensor([7]),
        torch.tensor([6]),
        torch.tensor([6]),
        torch.tensor([7]),
        torch.tensor([0])
    ]

    for test_cdf, expected_lo_idx, expected_hi_idx in zip(test_cdf_list, expected_lo_idx_list, expected_hi_idx_list):
        for test_device in ['cpu', 'cuda']:
            result = get_confidence_interval(torch.tensor(test_cdf).unsqueeze(-1).to(test_device), lower_cdf, upper_cdf)
            assert torch.all(result[0].to('cpu') == expected_lo_idx.to('cpu'))
            assert torch.all(result[1].to('cpu') == expected_hi_idx.to('cpu'))
