Project Title: “AI MUSIC RECOMMENDATION AND GENERATION”

Abstract :
AI-based playlist creation and music recommendation system is presented in this work. To find 
similarities between songs, the algorithm makes use of sensory characteristics and genre 
classifications obtained from music data. Through an analysis of the user's past listening choices 
or an existing playlist, the system suggests new music based on shared attributes. 
K-Means clustering is used in the recommendation process to classify songs according to their 
auditory characteristics. The songs in the dataset that are most comparable to the user's 
preferences are then found using a distance metric. With this method, the system may create 
playlists that suit the user's musical preferences.

Keywords: 
Music Recommendation System, Music Information Retrieval, K-Means 
Clustering, Audio Feature Analysis.

1. Introduction

1.1 Introduction

Human civilization has always been greatly influenced by music. It has the power to arouse a wide range of emotions, emotionally connect us, and transport us to 
many locations and times. The digital era has changed how we listen to music. We now have access to enormous song libraries at our fingertips because to the 
growth of music streaming services. This wealth of options, though, may sometimes be debilitating. Finding new music that suits our own tastes can be an 
arduous and time-consuming process.
Conventional approaches to finding music, including filtering by popularity or genre, might be restrictive. Genre classifications are frequently arbitrary and cover 
a broad spectrum of tones. Songs' originality or quality may not always be reflected in popularity numbers. 

1.2 Statement of the Problem

Information overload has become a problem for music consumers due to the growing amount of music accessible through streaming services. It could be difficult to sift 
through millions of music to discover new favourites. The recommendation systems that streaming providers currently provide are 
frequently based on basic algorithms that might not adequately represent each user's unique taste. These systems may prioritize recommending songs that are currently popular 
or that are similar to other songs by the same artist or genre, which might result in a monotonous listening experience. 
The goal of this project is to construct a more advanced AI-based playlist 

generation and music recommendation system in order to address this difficulty. The technology will make recommendations for new songs that are likely to 
connect with users based on their own musical inclinations.

1.3 Objectives
The principal aim of this project is to design a system for proposing music that utilizes artificial intelligence (AI) to generate customized listening experiences. 
The way the system will do this is by:

Comprehensive Music Analysis

Beyond simple genre categorization, the algorithm will examine more aspects of audio files, including instrumentals, danceability, energy, valence, and tempo. 
These elements offer a more impartial and advanced view of a song's musical qualities. 

Determining Song Similarities

Through the analysis of these audio characteristics, the system will be able to  recognize song links and patterns. This will enable the system to create a rich 
tapestry of musical links by grouping songs that have similar auditory characteristics. 

Creating User Profiles

By analysing a user's listening history, which includes the songs they most frequently listen to, the artists they follow, and the playlists they make, the system 
will be able to determine their musical preferences. 

Personalized Recommendations Generation 

The system will suggest new songs that are similar to the music the user already appreciates, based on the user's profile and the identified song relationships. With 
this method, the system may propose music that is actually customized to the user's tastes, going beyond basic genre recommendations.

Making Interactive Playlists

The playlists that the system creates will be customized to the unique needs and emotions of each user. The system may, for instance, generate relaxing 
music after an active day or an energizing playlist for working out.

2. Literature Survey
3. Algorithms for Producing Music Tackling the complex task of music generation, the research explores a range of algorithms in the machine learning and artificial intelligence fields. By utilizing 
methods like transformer models, generative adversarial networks (GAN), Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), and 
variational auto encoders (VAE), the research seeks to capture complex melodies and produce original works. CNNs are good at processing unprocessed audio data, 
whereas LSTM models are great at capturing long-range dependencies. 
Transformer models are experts at capturing sequential relationships in music  creation, whereas GANs and VAEs provide methods for producing varied and excellent music.

Limitations

Although these algorithms show great potential, they are not without flaws. Obstacles include potential overfitting with small datasets, managing long-term 
dependencies, and poor accuracy in capturing subtle musical expressions. Furthermore, it is yet difficult to include domain-specific musical information into 
the creation process, which affects how expressive and coherent songs are.

Algorithms Employed in Reinforcement Learning for Music Generation
Autonomous learning in music composition systems is clarified by investigating reinforcement learning techniques, such as Deep Q-Networks (DQN), Proximal 
Policy Optimization (PPO), Policy Gradient Methods, and Deep Deterministic Policy Gradient (DDPG). By using actor-critic models to balance exploration and 
exploitation, these techniques allow for the creation of nuanced music. However, issues with representing intricate musical structures, expressing subjective 
musical originality, and handling computational complexity continue to exist, which has an impact on the effectiveness and interpretability of generated music.

Limitations

The drawbacks of using reinforcement learning algorithms to create music highlight problems with computing complexity, modeling, sample efficiency, and 
interpretability. These algorithms have the potential to be useful, but they might have trouble capturing the complex and individualized aspects of musical creation, 
which could limit their application in practical settings.

Methodology for Literature Review and Data Classification
Study finds, categorizes, and evaluates pertinent publications using a methodical literature review approach. The goal of the research is to comprehend the 
distribution and features of the examined literature using an organized method that includes data classification according to publication type, institution, and 
geographic territory. However, restrictions on language use, the study's scope, and the possibility of missing pertinent works could affect how thorough the results
Limitations
The literature review process has limitations, such as the possibility of leaving out pertinent works and exclusions based only on language. Furthermore, limitations concerning the accessibility of unpublished works and the study's breadth in 
thoroughly covering various AI applications in music composition could impact the breadth of knowledge gained from the investigation.

5. Data Collection and Preprocessing
   
Description of the Dataset Used
Datasets used for AI MUSIC RECOMMENDATION AND GENERATION SYSTEM:
1. data.csv
2. data_by_artist.csv
3. data_by_geners.csv
4. data_by_years.csv
5. data_w_geners.csv
Purpose of dataset:
 Dataset is probably useful tool for researching music trends, popular artists, and genre traits.
 This data can be used for a variety of analysis and insights by industry experts, music Enthusiasts, and researchers.

Analysis of Dataset:
Each track's specific details are provided in the columns, including: 
Valence: The degree of positivity in music that a piece conveys. 
Year: The year the song was made available. 
Acousticness: A confidence metric indicating the acoustic nature of the track. 
Artist: The names of the musicians who were featured on the track. 
Danceability: A measure of a song's suitability for dancing, based on its steady 
speed and rhythm. 
Duration: musical piece is expressed in milliseconds (ms). 
Energy: Denotes fervor and movement; upbeat music has a quick, boisterous 
tone. 
Explicit Content: This field indicates whether or not the song has explicit 
material (0 for no, 1 for yes).
Instrumentalness: Indicates the degree of vocal presence. 
Key: Denotes the key in which the composition is composed. 
Liveness: Shows whether sounds from the audience are present. 
Decibels (dB): used to measure total loudness. 
Modality: Indicated by the mode (major as 1, minor as 0). 
Track Name: The song or piece's title. 
Popularity: Measures the song's popularity on a scale from 0% to 100%. 
Release Date: Indicates the actual year or date the song was released. 
Speechiness: Quantifies the amount of words that are said.
Tempo: Indicates the beats per minute (bpm) at which the song plays.

Data Cleaning and preprocessing steps
# Reading data
data = pd.read_csv(“data.csv”)
genere_data = pd.read_csv(‘data_by_generes.csv’)
year_data = pd.read_csv(‘data_by_year.csv’)
Code reads the csv files: “data.csv”, “data_by_geners.csv”, 
“data_by_years.csv”
Each dataset contains information about “music tracks, such as audio 
features, generes, and relese years”.
# Handling missing values
# Drop rows with NaN values
data = data.dropna()
# Drop rows with infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna()
The above code snippet can understand that we are droping null values at first. 
Then we are replacing positive and negative infinite infinite values to null and then droping them.

Dependent variable: popularity
Music over Time

Data is groped by year, we can understand how the overall sound has changed from 1921 to 2020.
Characteristics of Different Genres
This dataset contains the audio features of different songs along with the audio features of different genres. We can use this information to compare different 
genres and understand their unique differences in sound.

4. Methodology
4.1 Existing System
Traditional music recommendation systems frequently depend on basic algorithms that might not fully represent each user's unique tastes. These methods may 
prioritize suggesting songs that are currently popular or songs that are similar to one another by the same artist or genre.

These restrictions may result in:
Restricted Scope: Genre categories can be random and include a broad spectrum of musical qualities. 
Repeated Recommendations: Popularity measures might not accurately capture the uniqueness of a song and lead to unintresting listening sessions.

Proposed AI-based Music Recommendation System
This paper tackles the short comings of current systems by proposing an AI-based music recommendation system. Through the following features, the system uses 
machine learning and music data to provide

In-depth Music Analysis:
Examines a greater variety of auditory elements, such as danceability, energy, valence, pace, and instrumentals, in order to obtain a more impartial 
comprehension of a song's attributes.

Identifying Song Similarities:
The system groups songs with similar musical characteristics by using audio attributes to find patterns and correlations between songs.
Building User Profiles:
Gains insight into a user's taste in music by examining their listening history, which includes playlists they've made, artists they've followed, and songs they've enjoyed.

Generating Personalized Recommendations
Provides recommendations for new songs based on traits from music the user already likes, making the experience more tailored.

Creating Dynamic Playlists:
This feature creates playlists based on the individual requirements and moods of the user, such as a calming wind-down music or an energizing training playlist.
4.2 Proposed System Architecture
4.2.1 Architecture
Data Acquisition Module
Provides recommendations for new songs based on traits from music the user already likes, making the experience more tailored.
Machine Learning Module
This feature creates playlists based on the individual requirements and moods of the user, such as a calming wind-down music or an energizing training playlist.
Recommendation Module:
The recommendation module uses user profiles and music similarity data to provide tailored playlists and song recommendations.
User Interface Module
This offers a user interface via which users can communicate with the system, enter preferences, and get suggestions.
Data Import and Exploration:
A number of CSV files with music, genre, and year-grouped data are imported by the application. Next, it examines the data structure and looks for possible 
problems, such as missing values, using data.info().
Feature Correlation Analysis:
 To visualize data, the software imports the Yellowbrick library.
It describes salient aspects of a song's auditory qualities, such as danceability, 
energy, and valence.
 To investigate the relationship between these attributes and the desired variable (popularity), it generates a Feature Correlation visualizer. This makes it easier to 
comprehend how various audio qualities could affect a song's level of popularity.
Data Cleaning and Preprocessing:
In order to deal with missing values, the computer either removes rows that have NaN values or substitutes them with suitable techniques (such mean imputation).
13
For additional analysis, it specifies the target variable and feature names.
Data Understanding Through Visualization and EDA (Exploratory Data 
Analysis)
Music Over Time:
By charting attributes like danceability and energy and organizing data by year, the application examines how music has changed over time. This can show 
patterns in popular music genres over several decades.
Characteristics of Different Genres:
To comprehend how different genres differ in terms of sound characteristics, it analyzes audio variables between them (e.g., greater valence scores for pop music 
compared to rock).
Clustering Genres with K-Means:
The software groups genres according to their numerical acoustic properties using K-Means clustering. This aids in locating genre clusters.
4.2.2 Machine Learning Algorithms Used
K-Means Clustering
A clustering approach that can be used to find comparable music by grouping songs with similar audio qualities. By clustering songs into groups, helps the 
system to identify the similarities and patterns in the music data.
This helps in creating user’s profiles, personalized music recommendations, enhance user’s experiences and music recommendations.
Nearest Neighbors:
A method for locating comparable data points that may be utilized to suggest music that matches a user's taste. This can identify similarities and patterns 
among users. Users who have listed to similar songs or genres in the past are considered to be “nearest neighbours” in the feature space.
Matrix Factorization:
The fundamental idea behind this matrix factorization is to represent the user item interaction data, typically stored in a user-item matrix, as a product of two lower 
dimensional matrices, known as latent feature matrices.
User matrix: In this matrix, latent features are represented as columns and users as rows. 
A user's preferences across several latent features are represented by each row in the user matrix.
Item matrix: Music tracks are shown as rows in this item matrix, while latent features are shown as columns. The properties of a music track across several 
latent features are represented by each row of the item matrix.
4.2.3 Model Selection and Evaluation Metrics
It is likely that the system will select the particular machine learning models it 
uses based on criteria such as:
Data characteristics:
The type and format of the music data that is available are among the data characteristics.
Task requirements:
Whether to prioritize recommendation generating, user preference learning, or music similarity.
Performance metrics:
The efficiency of the selected models will be evaluated using metrics such as accuracy, precision, recall, and suggestion diversity.
Euclidean Distance: the distance between two points x and y in n-dimensional 
space is given by 
𝑑(𝑥, 𝑦) = √∑(𝑥𝑖 − 𝑦𝑖)

Nearest Neighbour: The nearest neighbor search locates the data point in X that is closest to query point q given a set of data points X and a distance measure 
(usually the Euclidean distance). The above-mentioned Euclidean distance formula is comparable to the method for determining the nearest neighbor.
Matrix Factorization: the user-item matrix R is broken down into two lower-rank matrices, U and V, whose products roughly equal R. Gradient descent and other 
optimization techniques are used to minimize the reconstruction error in order to approximate the result:
𝑅 ~ 𝑈𝑉2
Objective Function: In matrix factorization, the goal function is usually to minimize the Frobenius norm of the difference between R and its approximation, 
UV T.
𝑅 − 𝑈𝑉𝑡 2
𝐹
 where, F =Frobenius.
Experimental Setup 
5.1 Hardware and Software Used
The selection of hardware and software is essential for an AI music recommendation and generation system to ensure effective algorithm processing and implementation. 
Hardware 

CPU and GPU:
The computational demands of machine learning algorithms involve a strong CPU or GPU, particularly during the training and inference stages. Deep learning 
applications are particularly well-suited for GPUs due to their exceptional performance in parallel processing tasks.
RAM, or memory:
Large datasets must be efficiently stored and managed, which requires enough RAM. Large music datasets might require a lot of memory for feature extraction, 
pre-processing, and model training. 
Storage:
To store models for training, temporarily outcomes, and music datasets, highcapacity storage disks are required. Speedier read/write rates are provided by solid-state drives
(SSDs) in comparison to conventional hard disk drives (HDDs), enabling speedier data access.

Software:
Python 
Python's large libraries and signal processing and machine learning frameworks make it the main programming language used to create AI music 
recommendation and generating systems. NumPy, Pandas, and Scikit-learn are libraries that are frequently used for training models and manipulating data.
Machine Learning Libraries:
For the purpose of implementing machine learning and deep learning models, libraries like Tensor Flow, PyTorch, and Keras offers strong tools. These libraries 
provide pre-configured modules for assessment metrics, optimization techniques, and neural network topologies.
Development Environments:
Popular development environments for AI projects that provide interactive computing and collaborative capabilities are Kaggle, Google Colab, and Jupyter 
Notebook. These platforms give customers access to GPU-accelerated computing resources, which facilitates the effective training of complicated models.
Database Management Systems (DBMS):
User preferences, recommendation history, and music information can be stored and managed using DBMS platforms such as SQLite, MySQL, or PostgreSQL. These 
systems offer effective methods for storing and retrieving data, making them suitable for managing substantial datasets.
Visualization Tools:
Data visualization and analysis are done using visualization packages such as Matplotlib and Seaborn. With the use of these tools, users may clearly and 
understandably view music attributes, model predictions, and recommendation outcomes.
5.2 Parameter Tuning Process

Optimizing the k-means clustering algorithm's parameters to improve the system's accuracy and performance is known as the "parameter tuning process" 
for the AI music recommendation and generation system. The goal of this procedure is to determine the ideal values for several parameters, including the 
distance metric and the number of clusters (k). 
Number of Clusters (k):
The number of clusters (k) in the k-means method is one of its important parameters. The efficacy and granularity of the clustering process are determined 
by the ideal value of k. Several methods, including the elbow method, silhouette score, and silhouette analysis, can be used to adjust this value. These methods 
assist in determining the value of k that optimizes intra-cluster similarity while minimizes inter-cluster similarity by analyzing the clustering results for various 
values of k.
Distance Metric:
The clustering findings are influenced by the distance metric that is selected to determine the degree of similarity between the data points. Cosine similarity, 
Manhattan distance, and Euclidean distance are examples of common distance measures. The right distance metric must be chosen based on the goals of the 
clustering process and the type of music data. In order to fine-tune the distance metric, one must test out several metrics and assess how they affect the 
performance of clustering.
Initialization Method:
The initialization method, which establishes the initial location of cluster centroids, is an additional parameter in k-means clustering. The clustering algorithm's 
stability and convergence may be impacted by the initialization technique selected. K-means++ initialization and random initialization are two popular 
initialization techniques. Comparing the effectiveness of several starting techniques and choosing the one that produces the most reliable and accurate 
clustering results is known as parameter tweaking.
Convergence Criteria:
The conditions under which the k-means algorithm breaks are specified by the  convergence criterion. The process usually repeats until the centroids stabilize or 
until a predetermined number of iterations is reached. A few examples of parameters to tweak include the maximum number of iterations and the centroid 
movement threshold while tuning the convergence criteria. This prevents overfitting and early algorithm termination while guaranteeing an efficient 
convergence of the algorithm.
Evaluation Metrics:
Evaluation measures like the pattern score, are used to gauge the quality of the clustering results throughout the parameter tweaking phase. By offering 

quantifiable assessments of clustering performance, these metrics make it possible to compare various parameter combinations in an unbiased manner.
7. Results and Discussion
Clustering Genres with K-means
K-means clustering is used to divide the genres in this dataset into ten clusters based on the numerical audio features of each genres. This t-Distributed 
Stochastic Neighbor Embedding (t-SNE) is used to reduce dimensionaility of the genre data and visualize it in a two-dimensional space.

This uses PCA to reduce dimensionality and Plotly Express to clusters in a twodimensional space after K-means clustering is applied to the song data and cluster 
labels are assigned to each song.

Output:
It is evident from the analysis and visualizations that comparable song types are clustered together, and that comparable genres also contain data points that are 
positioned close to one another.
Songs within related genres will sound similar and originate from similar eras. Similarly, genres will sound similar. By using the data points of the songs a user 
has listened to, we may leverage this concept to create a recommendation system that suggests songs based on adjacent data points.
Suggested Songs:
A list of songs is provided by the system, together with information about the artists, albums, and years of release.
Code Excerpt: 
A snippet of Python code output can be seen in the image below. 
It defines a function named recommend_song, whose input is a list of dictionaries, most likely containing song data. 
Based on user preferences, the function most likely generates song recommendations. 
Fig (12): Output showing AI Music Recommendation and Generation.

Conclusion and Future Scope
Conclusion
The AI-based system for creating playlists and recommending music seems promising. It uses sensory features and music genres to find similarities between 
songs and suggest music based on common traits. The system also uses K Means clustering to group songs based on how they sound. However, it might struggle 
to capture all the unique aspects of music creation, which could affect how well it works. To improve, the system could consider user feedback, use better machine 
learning techniques, and refine how it recommends music. This could help make the music listening experience more personalized and varied for users in the 
future.
Future Scope
The AI-based music recommendation and playlist creation model have a bright future. To make it even better, we can add sentiment analysis to suggest music 
based on how users feel. Using advanced machine learning like GANs and VAEs could help create more diverse music, appealing to more people. It would also 
help to use real-time data to keep up with current music trends and what users like. Adding social features would let users share and find music together, making 
the experience more fun and interactive. By doing these things, the model can become a great music platform that not only suggests personalized music but also 
helps people make music and connect with others.

References
 Born, G., Morris, J., Diaz, F., & Anderson, A. (2021). Artificial intelligence, music recommendation, and the curation of culture.
 Afchar, D., Melchiorre, A., Schedl, M., Hennequin, R., Epure, E., & Moussallam, M.(2022). Explainability in music recommender systems. AI Magazine, 43(2), 190-
208.
 Li, J., & Li, S. (2011). Music Recommendation Using Content-Based Filtering and Collaborative Filtering with Data Fusion Approach. *International Conference on 
Advanced Data Mining and Applications.
 Kim, S., & Lee, J. H. (2015). Music Recommendation Based on Emotional Cues. International Conference on Music Information Retrieval.
 Lian, J., & Gou, L. (2013). Combining Collaborative Filtering and Sentiment Analysis for Music Recommendation. IEEE International Conference on Big Data.
 Andjelkovic, I., D. Parra, and J. O’Donovan. 2016. “Moodplay: Interactive Moodbased Music Discovery and Recommendation.” In Proceedings of the 2016 
Conference on User Modeling Adaptation and Personalization, 275–9. New York: ACM
 Abdollahi, B., and O. Nasraoui. 2016. “Explainable Matrix Factorization for Collaborative Filtering.” In Proceedings of the 25th International Conference Companion on World Wide Web, 5–6
 Goto, M., and R. B. Dannenberg. 2019. “Music Interfaces based on Automatic 
Music Signal Analysis: New Ways to Create and Listen to Music.” IEEE Signal Processing Magazine 36(1): 74–81
