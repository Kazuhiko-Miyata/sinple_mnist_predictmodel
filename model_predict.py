import torch
from sklearn.model_selection import train_test_split

digit_images = digits_data.data
labels = digits_data.target
x_train, x_test, t_train, t_test = train_test_split(digit_images, labels)#25%がテスト用

#Tensorに変換
x_train = torch.tensor(x_train, dtype=torch.float32)
#入力:訓練用
t_train = torch.tensor(t_train, dtype=torch.int64)
#正解:訓練用
x_test = torch.tensor(x_test, dtype=torch.float32)
#入力:テスト用
t_test = torch.tensor(t_test, dtype=torch.int64)
#正解:テスト用

#モデルの構築
from torch import nn

net = nn.Sequential(
    nn.Linear(64, 32),#全結合層
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)
print(net)

#学習
from torch import optim

#ソフトマックス関数+交差エントロピー関数
loss_fnc = nn.CrossEntropyLoss()

#SDG　モデルのパラメータを渡す
optimizer = optim.SGD(net.parameters(), lr=0.01) #学習率は0.01

#損失のログ
record_loss_train = []
record_loss_test = []

#訓練データを1000回使う
for i in range(1000):
  #パラメータの勾配を0に
  optimizer.zero_grad()

  #順伝播
  y_train = net(x_train)
  y_test = net(x_test)

  #誤差を求めて記録する
  loss_train = loss_fnc(y_train, t_train)
  loss_test = loss_fnc(y_test, t_test)
  record_loss_train.append(loss_train.item())
  record_loss_test.append(loss_test.item())

  #逆伝播
  loss_train.backward()

  #パラメータの更新
  optimizer.step()

  if i%100 == 0: #100回ごとに経過を表示
      print("Epoch:", i, "Loss_Train:", loss_train.item(), "Loss_Test:", loss_test.item())

#訓練済みモデルを使って予測
img_id = 0
x_pred = digit_images[img_id]
image = x_pred.reshape(8, 8)
plt.imshow(image, cmap="Greys_r")
plt.show()

x_pred = torch.tensor(x_pred, dtype=torch.float32)
y_pred = net(x_pred)
print("正解率:", labels[img_id], "予測結果:", y_pred.argmax().item())
