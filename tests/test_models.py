from cptlms.models.prompt_encoder import PromptEncoder


def test_prompt_encoder_emb():
    t, v, h = 2, 4, 6
    prompt_encoder = PromptEncoder(
        token_dim=t,
        num_virtual_tokens=v,
        hidden_size=h,
        reparam_type="emb",
    )

    emb = prompt_encoder.forward()
    assert emb.shape == (1, v, t)


def test_prompt_encoder_mlp():
    t, v, h = 2, 4, 6
    prompt_encoder = PromptEncoder(
        token_dim=t,
        num_virtual_tokens=v,
        hidden_size=h,
        reparam_type="mlp",
    )

    emb = prompt_encoder.forward()
    assert emb.shape == (1, v, t)


def test_prompt_encoder_lstm():
    t, v, h = 2, 4, 6
    prompt_encoder = PromptEncoder(
        token_dim=t,
        num_virtual_tokens=v,
        hidden_size=h,
        reparam_type="lstm",
    )

    emb = prompt_encoder.forward()
    assert emb.shape == (1, v, t)
