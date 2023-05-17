**Project Documentation: Rekko Movie Recommendation Engine**

**Overview**

Rekko is a movie recommendation engine designed to provide personalized film suggestions based on user interactions and film metadata. It uses information from the user's watched movies, the length of time spent watching each film, and external API data including average movie ratings and descriptions. The engine calculates cosine similarity scores between movies to suggest films that align closely with the user's preferences.

The entire application is containerized using Docker, making it easy to set up and run on any platform.

**Prerequisites**

Docker
Python 3.7 or higher

**Input Data**

The Rekko engine requires two dataframes as input:

_interactions_ - This dataframe contains data about user interactions with films. The primary feature used for determining whether a user likes a movie is calculated as the time they watched the film divided by the length of the film.

_movies_md_ - This dataframe contains metadata about the movies.

**External API**

Rekko parses an external API to fetch additional information about each movie, including its average rating and description. This information is also factored into the recommendation score.

**Cosine Similarity Matrix**

After gathering all necessary data, Rekko calculates a cosine similarity matrix for all movies. This matrix is used to identify movies that are most similar to those that the user has watched and liked.

**Pipeline**

When we have all the data, we need to run the pipeline which performs feature engeneering and tuning of the data frames. It consists of a list of functions from model.py and result in clean version of data to be used further in the model.

**Output**

The output of the Rekko engine is a list of movies, each associated with a score. The higher the score, the more likely the user is to enjoy that movie.

**Setup**

Run pipeline code:
docker run -p 4000:80 -e P1=’ data/interactions.parquet’ -e P2=’ data/movies_metdata.parquet’ -e P3=’data/Movies_r_d.xlsx’ -e P4=’data/movies_rd.parquet’ -e P5=’ data/int_coef.parquet’ -e P6=’ data/movies_md_upd.parquet’ -e P7=’data/interactions_upd.parquet’ -e P8=’matrix/cos_sim_md.pkl’ -e P9=’ matrix/cos_sim_desc.pkl’

Run entrypoint code:

docker run -p 4000:80 -e P1=*user id* -e P2=*number of rekkos* -e P3='data/interactions_upd.parquet' -e P4='data/movies_md_upd.parquet' -e P5='data/int_coef.parquet' -e P6='matrix/cos_sim_md.pkl' -e P7='matrix/cos_sim_desc.pkl' rekko

The application will be available at localhost.
