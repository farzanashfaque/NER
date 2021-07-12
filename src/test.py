dic = {
    'per': ['joe', 'ando', 'hirsh']
}

@staticmethod
def active_loss(y_true, y_pred):
    idx = y_true[:,:,0]!=1
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    lfn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction='none'
    )
    loss = K.mean(lfn(y_true, y_pred), axis=-1)
    return loss

for key,val in dic.items():
    with open(f'../predictions/{key}.txt', 'w') as file:
        for entity in val:
            file.write(entity)
        file.close()