import random
import re
import math
import os.path

def check_file(file_name):
    return os.path.isfile(file_name)

def preprocessing(data):
    tweets = [line.strip('\n')[50:] for line in data]
    tweets = [re.sub(r"http\S+|www\S+", "", tweet) for tweet in tweets]
    tweets = [''.join(filter(lambda x: x not in string.punctuation, tweet.lower())) for tweet in tweets]
    return [tweet.split() for tweet in tweets]

def k_means(tweets, k=4, max_iterations=50):
    centroids = random.sample(tweets, k)
    prev_centroids = []

    while prev_centroids != centroids and max_iterations > 0:
        clusters = assign_clusters(tweets, centroids)
        prev_centroids = centroids
        centroids = update_centroids(clusters)
        max_iterations -= 1

    sse = SSE(clusters)
    return clusters, sse

def assign_clusters(tweets, centroids):
    clusters = {i: [] for i in range(len(centroids))}
    for tweet in tweets:
        min_distance = math.inf
        cluster_index = -1
        for i, centroid in enumerate(centroids):
            distance = Jaccard_distance(centroid, tweet)
            if distance < min_distance:
                min_distance = distance
                cluster_index = i
        clusters[cluster_index].append([tweet, min_distance])
    return clusters

def update_centroids(clusters):
    centroids = []
    for cluster in clusters.values():
        min_sum_distance = math.inf
        centroid_index = -1
        for i, (tweet, _) in enumerate(cluster):
            sum_distance = sum(Jaccard_distance(tweet, t[0]) for t in cluster)
            if sum_distance < min_sum_distance:
                min_sum_distance = sum_distance
                centroid_index = i
        centroids.append(cluster[centroid_index][0])
    return centroids

def Jaccard_distance(tweet_1, tweet_2):
    t1, t2 = set(tweet_1), set(tweet_2)
    intersection = len(t1.intersection(t2))
    union = len(t1.union(t2))
    return 1 - (intersection / union)

def SSE(clusters):
    sse = 0
    for cluster in clusters.values():
        sse += sum(distance ** 2 for _, distance in cluster)
    return sse

def main():
    file_name = input("Enter file name: ")
    if not check_file(file_name):
        print("File does not exist.")
        return

    try:
        with open(file_name, encoding='utf8') as f:
            tweets = preprocessing(f)
    except UnicodeDecodeError:
        with open(file_name, encoding='cp1252') as f:
            tweets = preprocessing(f)

    experiments = 5
    k = 3

    for e in range(experiments):
        print(f"Experiment {e + 1} for k = {k}:")
        clusters, sse = k_means(tweets, k)
        for i, cluster in enumerate(clusters.values()):
            print(f"Cluster {i + 1}: {len(cluster)} tweets")
        print(f"SSE: {sse}\n")
        k += 1

if __name__ == "__main__":
    main()
