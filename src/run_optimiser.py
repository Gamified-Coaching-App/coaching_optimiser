from environment.environment import Environment
from contextual_bandit.contextual_bandit import ContextualBandit
from RunningDataset.RunningDataset import RunningDataset

def train_optimiser(): 
    preprocessor = RunningDataset()
    X_train = preprocessor.preprocess()
    env = Environment(X_train)
    bandit = ContextualBandit(state_shape=(21,10), action_shape=(7,7), env=env)
    epochs = 2
    batch_size = 1000

    for epoch in range(epochs):
        for index in range(0, len(X_train), batch_size):
            state_batch = env.get_states(slice(index, index +batch_size))
            reward = bandit.train(state_batch)
            print(f"Epoch {epoch + 1}, Batch starting at {index + 1}, Average reward: {reward}")
    
    bandit.model.export('../model/optimiser')

if __name__ == "__main__":
    train_optimiser()