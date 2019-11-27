import gym
import numpy as np
import os

if 0:
    # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    import keras as keras
    from keras.layers import Dense
    from keras.models import Sequential
else:
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential



# create the model for the policy
model = Sequential()
model.add(Dense(units=64, input_dim=4))#, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(units=64))#, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(units=1, activation='sigmoid'))#, kernel_initializer='RandomNormal'))

optimiser = keras.optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])

model.summary()
print (model.get_config())


# set up env
env = gym.make("CartPole-v0")
observation = env.reset()

LEFT_ACTION = 0
RIGHT_ACTION = 1

gamma = 0.99

x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_nb = 0
history = []


# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r, gamma):
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    # we go from last reward to first one so we don't have to do exponentiations
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


# run the simulation and learn the policy
while (True):
    x = observation

    proba = model.predict(np.expand_dims(x, axis=1).T)
    p = np.random.uniform()
    action = RIGHT_ACTION if proba > p else LEFT_ACTION

    x_train.append(x)
    y_train.append(action)

    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward

    if done:
        print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)
        history.append(reward_sum)
        episode_nb += 1

        # training
        dw = discount_rewards(rewards, gamma)
        # print (dw, rewards)
        model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=0, sample_weight=dw)#, epochs=1)

        # Reinitialization
        x_train, y_train, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0

        if episode_nb > 100 and np.mean(history[-100:]) >= 195.0:
            print('** SOLVED ** in ', episode_nb)
            from datetime import datetime
            model_name = 'cartpole-vpg-%s.h5'%datetime.now().strftime("%Y%m%d-%H%M%S")
            model.save_weights(model_name)
            break

        if episode_nb > 50 and np.mean(history[-50:]) < 10.0:
            print('** CRASHED ** in ', episode_nb)
            print (dw, rewards)
            print(proba, action, p)
            # break
        # if episode_nb > 300: break

