# Context

Question: Using the clustering models learned in ENSF 611, experiment and explore how effective each of them are in clustering clues from the game show Jeopardy! based on the keywords used in each clue. If the models learned in class are ineffective, see if there are any models that do work.

Motivation: Whenever I watch the game show Jeopardy!, I have always been curious as to which categories and topics appear the most frequently in clues that appear on the show.

Looking at the fan-created archive website J! Archive (https://j-archive.com/), which contains a database of all the clues aired since the debut of the show in 1984, I noticed a definite shift in the writing style of the clues and category names from then to now.

For example, it would be easy to know what the category of an Opera clue is from an episode in 1984, since the category at the top is shown to be “Opera”. However, in a 2021 Opera clue example this is not as clear, since the category is shown as “Aria Grande” – this is a play on the name of the pop singer Ariana Grande and also references the opera term “Aria”, which is self-contained opera piece for one voice. The category name hints at indicates multiple possibilities for what the clue may be about. The clue content in the 2021 example is also less straightforward and contains trickier writing, making the answer harder to find.

The main point is that the category and content of the clues that appear on the current episodes of Jeopardy! are less obvious and more subtle. Therefore, choosing to separate clues based solely on category name may not be the most reliable way to find out which topics appear the most frequently in clues.

Instead, what if we clustered clues based on keywords used in the clue instead? Would this improve the chances of discovering distinct topics for the clues?

Expectations: As someone who is new to the topic of Machine Learning, I definitely do not expect the clustering models I choose to produce worthwhile results, especially since the clue language is so complex and tricky. This is more of an experimental project for my own curiosity to see first-hand how effective models can be with real-world data.

# Background on Code Files
Download the entire project folder and open the Jupyter notebook file ensf-611-final-project.ipynb, which contains all of the code I wrote for this project. 

Please run each code block in order, one at a time. A few code blocks may take longer than others (5-10 minutes) - if this is the case, please wait for the code block to finish executing before moving on to the next one. Descriptions have added throughout the file to keep you informed of what is happening. No extra packages or libraries should be needed - a code block will import a package if it is needed for a task.

# Summary and Interpretation

To recap, we have the following validation metrics for each model:
- Our K-Means model produced a silhouette score of 0.779, a Calinski-Harabasz score of 747317.65, and a Davies-Bouldin index of 0.52.
- Our Gaussian Mixture model produced a silhouette score of 0.636, a Calinski-Harabasz score of 301526.41, and a Davies-Bouldin index of 1.48.
- Our DBSCAN model produced a silhouette score of 0.802, a Calinski-Harabasz score of 30609.66, and a Davies-Bouldin index of 1.43.
- Our Agglomerative model produced a silhouette score of 0.546, a Calinski-Harabasz score of 10498.91, and a Davies-Bouldin index of 0.67.

Overall, our K-Means model has the best validation matrics - the silhouette score of 0.779 indicates highly dense clustering and is excellent, especially considering that I am using the entire dataset. The Calinski-Harabasz score of 747317.65 is by far the highest-ranked and K-Means also has the best Davies-Bouldin index at 0.52 (closest to 0) which indicates a better partition and less similarity compared to the other models. From manually checking a cluster, it is encouraging that the cluster had questions on various types of history and literature.

Our Gaussian Mixture model produced one of the lowest silhouette scores, but at 0.636 it is still a quality cluster since it is over 0.5, especially considering that the entire dataset is being used. The model had the second-best Calinski-Harabasz score at 301526.41, indicating that the model has better defined clusters compared to DBSCAN and Agglomerative. The Davies-Bouldin index of 1.48 was the lowest though, indicating that the clusters are somewhat similar. From manually checking a cluster, I was glad to see that a decent number of clues were based off questions with "body" or "bodies" as a keyword, which resulted in anatomy or bodies of water categories being prominent.

The DBSCAN model produced the best silhouette score at 0.802, the third-best Calinski-Harabasz score at 30609.66, and the second-worst Davies-Bouldin score of 1.43. Keep in mind that these results were produced with a sample of 50000 questions instead of the full dataset, due to time and memory contraints. Manually, the questions from the cluster I checked did not look similar - the topics and official categories are quite varied. The only thing I could possibly say are similar are the length of the questions - they seem shorter, direct, and to-the-point. The manual check of the DBSCAN cluster did not really correlate with the silhouette score results that were produced.

Finally, the Agglomerative model produced the worst results overall (despite only having a sample of 10000) as it had the worst silhouette score at 0.546 (but still a decent number since it is over 0.5), the worst Calinski-Harabasz score by far at only 10498.91, and a decent Davies-Bouldin index at 0.67. Also, the two clusters I checked looked quite similar - both were quite varied as Sports were the leading categories for both clusters, but not by much. There seems to be a little bit of everything in each cluster, and there wasn't much distinguishing the two.

Using these results, let's interpret my original question from my project proposal: Using the clustering models learned in ENSF 611, experiment and explore how effective each of them are in clustering clues from the game show Jeopardy! based on the keywords used in each clue.

Answer: Although none of the models produced particularly amazing results, I would say the K-Means model and Gaussian Mixture models produced the best results overall based off their validation metrics and manual checks that I talked about above. I am curious if it is possible that DBSCAN and Aggomerative could produce better results, given a larger sample.

# Reflection

Why Did I Select This Problem to Solve?

I wanted to explore if I could cluster clues from the game show Jeopardy! based off their keywords, instead of relying solely on the clue category to determine which topics appear the most frequently in clues. I wondered if separating clues based on the words would be more reliable for finding distinct clue topics than relying on just category names, which have become more cryptic as time has passed. This was also an experimental project for my own curiosity, to see first-hand how effective the clustering models I learned in class can be with clustering real-world data.

Deviations from my Proposal?
- Instead of using three versions of the clues CSV file (all clues from 1984-2021, clues from 1984-1995, and clues from 2010-2021) I just used the original version with all clues from 1984-2021 due to time constraints. The code I wrote in this file had become quite long and I didn't want to repeat it for two other datasets.
- As I mentioned, the code in this notebook file had become quite long - much longer than I expected. For these reasons I did not explore other clustering models like spectral clustering or Doc2Vec like I planned - these are bonus tasks I can revisit in the future.
- I did not expect to manually check questions and categories from random clusters for each model and visualize them in a pie/donut chart, but thought it was a great idea instead of just looking at validation metrics.

Project Difficulties and What I Learned?

The most difficult parts of this project included finding the appropriate parameters to use for each model - I learned about using the Elbow method to find the appropiate amount of clusters for K-Means model, plotting the Akaike information criterion (AIC) and the Bayesian information criterion (BIC) to find the appropiate amount of components for the Gaussian Mixture model, and using a Dendrogram to find the appropiate amount of clusters for Agglomerative model. It was also difficult figuring out a way to compare each centroid center from the K-Means model to each data sample, and needed to search code written by someone (https://medium.com/@williamsuh/unsupervised-learning-based-on-jeopardy-questions-part-2of-3-68c18c3490bd) who had already done it. A lack of memory was also an issue, as training models and computing silhouette scores took a long time. For DBSCAN and Agglomerative clustering, I had to use a small sample size because there wasn't enough memory to fit the entire dataset. 

Given extra time and resources, I would perform a grid search to determine the best parameters to use, especially for DBSCAN. I would also use a more powerful computer or parallel computing in order to fit the DBSCAN and Agglomerative models to the entire dataset. I would also explore other clustering models like Doc2Vec or spectral clustering.

This is my first time doing a machine learning project, and I really enjoyed being able to apply the concepts I learned in class to real-world data, regardless of the results. The easiest and most enjoyable parts of this project were the data exploration, data cleaning, and data visualization aspects.