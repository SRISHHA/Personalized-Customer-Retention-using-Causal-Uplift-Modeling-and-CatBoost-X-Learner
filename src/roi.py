def compute_roi(uplift, discount_cost=100, revenue=1000):

    expected_gain = uplift * revenue
    roi = expected_gain - discount_cost

    return roi


def decision(uplift):
    if uplift > 0.2:
        return "Offer Discount"
    elif uplift > 0:
        return "Low Priority Offer"
    else:
        return "Do Not Target"