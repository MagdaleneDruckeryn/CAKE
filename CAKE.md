```python
以下为初次生成的结果
```


```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class PianoCAE(Model):
    def __init__(self, input_shape=(256, 256, 3), condition_dim=5):
        super(PianoCAE, self).__init__()
        # 复用你提供的 ResNet50 Encoder
        self.encoder_backbone = build_encoder_backbone(input_shape)
        
        # 条件注入层：处理 pitch, velocity 等参数 [cite: 50]
        self.condition_dense = layers.Dense(64, activation='relu')
        
        # 解码器：将特征还原为频谱图 [cite: 52]
        self.decoder = tf.keras.Sequential([
            layers.Dense(8*8*256, activation='relu'),
            layers.Reshape((8, 8, 256)),
            layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid') # 还原至 256x256
        ])

    def call(self, x, condition):
        # 1. 提取声纹特征
        latent = self.encoder_backbone(x)
        # 2. 嵌入外部条件参数 [cite: 60]
        cond_emb = self.condition_dense(condition)
        # 3. 拼接特征与条件 (CAE 核心逻辑)
        combined = tf.concat([latent, cond_emb], axis=-1)
        # 4. 重建频谱
        return self.decoder(combined)
```
