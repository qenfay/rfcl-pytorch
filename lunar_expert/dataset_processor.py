import minari


dic = minari.list_local_datasets()

for (k,v) in dic.items():
    print(f"dataset name: {k}, metadata: {v}\n")
    
dataset = minari.load_dataset("lunarLander/reverseCurriculum-v0")

print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)

episodes = dataset.sample_episodes(n_episodes=1)
