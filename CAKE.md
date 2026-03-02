<h3 style="font-family: '田氏颜体大字库',serif; font-weight: light;">第一部分：数据加载管道</h3>
<h3 style="font-family: '田氏颜体大字库', serif; font-weight: light;">模型需要同时接收声纹图与特征参数。使用 tf.data.Dataset 构建高吞吐量数据管道。</h3>


```python
import tensorflow as tf

def parse_data(tfrecord_proto):
    # 解析TFRecord中的声纹图与特征向量
    pass

def build_dataset(tfrecord_paths, batch_size=16):
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    dataset = dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

<h3 style="font-family: '田氏颜体大字库',serif; font-weight: light;">第二部分：自定义损失函数</h3>
<h3 style="font-family: '田氏颜体大字库', serif; font-weight: light;">模型需要确保解码后的声纹图与原图一致。这里使用均方误差衡量重建损失。</h3>

<img width="596" height="320" alt="5f9f8aaa-2f8a-429a-a465-8ec1427f85d0" src="https://github.com/user-attachments/assets/5fc14910-8007-45b1-a419-133401f04b78" />

<h3 style="font-family: '田氏颜体大字库',serif; font-weight: light;">第三部分：自定义训练循环</h3>
<h3 style="font-family: '田氏颜体大字库', serif; font-weight: light;">实现对多输入的梯度计算与参数更新。</h3>


```python
class PianoCAE(Model):
    # 此处保留之前定义的 __init__ 和 call 方法

    def train_step(self, data):
        # 解包数据：x为原始声纹图，condition为控制参数，y为目标声纹图
        x, condition, y = data

        with tf.GradientTape() as tape:
            predictions = self(x, condition, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}
```

<h3 style="font-family: '田氏颜体大字库',serif; font-weight: light;">第四部分：编译与执行</h3>
<h3 style="font-family: '田氏颜体大字库', serif; font-weight: light;">实例化模型并启动训练。</h3>


```python
model = PianoCAE()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.MeanSquaredError()
)

# 挂载数据集并启动训练
# train_dataset = build_dataset(['data.tfrecord'])
# model.fit(train_dataset, epochs=50)
```
