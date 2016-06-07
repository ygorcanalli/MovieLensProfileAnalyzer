using Recsys
using DataFrames
reload("Recsys")

function createModel(data)
  return Recsys.RegularedSVD(data);
end

genres = [:unknown, :action, :adventure, :animation, :childrens, :comedy, :crime, :documentary, :drama, :fantasy, :film_noir, :horror, :music, :mystery, :romance, :sci_fi, :thriller, :war, :western]
header = [:movie_id, :title, :release_date, :video_release_date, :imdb_url]
movies = readtable("ml-100k/u.item", header = false, separator = '|', names = vcat(header,genres))

function recommend(model, user_id, dataset)
  all_movies = unique(movies[:movie_id])
  view_movies = dataset[dataset[:user] .== user_id, :][:item]
  # remove view movies from list of movies to predict
  deleteat!(all_movies, findin(all_movies, view_movies))
  data = ones(Int32, (size(all_movies)[1], 2))
  data[:,1] = data[:,1] * user_id
  data[:,2] = all_movies
  predicted = model.predict(data)
  titles = movies[all_movies, :title]
  predicted = hcat(all_movies, predicted, titles)
  predicted = sortrows(predicted, rev=true, by=x->(x[2]))

  df = DataFrame()

  df[:item] = predicted[:, 1]
  df[:prediction] = predicted[:, 2]
  df[:title] = predicted[:, 3]

  return df
end

function contamine_dataset(dataset, user_id, contamine_list, n)
  rows_created = 0

  for i = 1:size(contamine_list, 1)

    # get the rating of user on that movie, if exists
    mask = (dataset.file[:user] .== user_id ) & (dataset.file[:item] .== contamine_list[i])
    target = dataset.file[mask, :]

    # if user hasn't seen the movie
    if size(target, 1) == 0
      # create a new rating of 5 stars
      push!(dataset.file, [user_id, contamine_list[i], 5, 0])
      rows_created += 1
    end

    # when there's n new rows, stop
    if rows_created == n
      break
    end

  end

  return dataset
end

function noisy_recommend(user_id, contamine_list, n)
  contamined_ratings = Recsys.Dataset();
  contamined_ratings = contamine_dataset(contamined_ratings, user_id, contamine_list[:item], n)
  contamined_model = createModel(contamined_ratings);
  return recommend(contamined_model, user_id, contamined_ratings.file)
end

function get_popular_movies(ratings)

  good_ratings = ratings.file[ratings.file[:rating] .> 3, :]
  rename!(good_ratings, :item, :movie_id)
  # sort movies by number of good ratings
  popular_movies = join(movies, good_ratings, on = :movie_id, kind = :inner)
  rename!(popular_movies, :movie_id, :item)
  popular_movies = by(popular_movies, [:item, :title], nrow)
  rename!(popular_movies, :x1, :popularity)
  sort!(popular_movies, cols = [:popularity, :item], rev = true)

  return popular_movies
end

function by_category(dataset, category)
  category_movies = movies[movies[category] .== 1, :movie_id]
  return dataset[find(x->in(x,category_movies), dataset[:item]), :]
end

function dcg(ranking)
  sum = 0
  for i = 1:size(ranking, 1)
    sum += (2^ranking[i, :prediction] - 1)/log2(i+1)
  end

  return sum
end

function ndcg_at_k(correct, predicted, k)
  iDCG_k = dcg(correct[1:k, :])
  DCG_k = dcg(predicted[1:k, :])

  nDCG_k = DCG_k/iDCG_k
  return nDCG_k
end

function precision_at_k(correct, predicted, k)
  relevants_retrieved = size(findin(predicted[1:k, :item], correct[1:k, :item]), 1)
  relevants = k
  return relevants_retrieved / relevants
end

function mae(correct, predicted)
  sum = 0
  for i = 1:size(correct)[1]
    sum += abs(correct[i] - predicted[i])
  end
  return sum
end

function rmse(correct, predicted)
  sum = 0
  for i = 1:size(correct)[1]
    sum += (correct[i] - predicted[i]) ^ 2
  end
  return sqrt(sum)
end



# Hipótese: Alto impacto ao adicionar um item que o usuário não gosta nas previsões futuras
function hypothesys_1(user_id, category, noise_sizes)
  ratings = Recsys.Dataset();
  popular_movies = get_popular_movies(ratings)

  model = createModel(ratings);
  recommendations = recommend(model, user_id, ratings.file)

  popular_by_category = by_category(popular_movies, category)

  noisy_recommendations = []
  #ndcg_at_20 = []
  precisions_at_k = []
  for i in 1:length(noise_sizes)
    nr = noisy_recommend(user_id, popular_by_category, noise_sizes[i])
    push!(noisy_recommendations, nr)

    #ndcg = ndcg_at_k(recommendations, nr, 20)
    #push!(ndcg_at_20, ndcg)

    precisions = []
    for j = 1:20
      push!(precisions, precision_at_k(recommendations, nr, j))
    end
    push!(precisions_at_k, precisions)
  end

  result = Dict("recommendations" => recommendations, "noisy_recommendations" => noisy_recommendations, "precisions_at_k" => precisions_at_k)
  return result
end

function hypothesys_2_alpha(user_id, category, margin, noise_size)
  ratings = Recsys.Dataset();

  category_movies = movies[movies[category] .== 1, :]
  shuffle_index = shuffle([1:size(category_movies, 1)])

  test_movies_index = find(r -> r >= size(category_movies, 1) * margin, shuffle_index)
  test_movies = animation_movies[test_movies_index, :]

  test_ratings_index = find(r-> in(r, test_movies[:movie_id]), ratings.file[:item])
  test_ratings = ratings.file[test_ratings_index, :]

  train_ratings = Recsys.Dataset();
  deleterows!(train_ratings.file, test_ratings_index)

  popular_movies = get_popular_movies(train_ratings)
  popular_by_category = by_category(popular_movies, category)

  contamine_dataset(train_ratings, user_id, popular_by_category[:item], noise_size)

  noisy_model = createModel(train_ratings);
  noisy_recommendations = recommend(noisy_model, user_id, train_ratings.file)
  test_noisy_recommendations_index = find(r-> in(r, test_movies[:movie_id]), noisy_recommendations[:item])
  test_noisy_recommendations = noisy_recommendations[test_noisy_recommendations_index, :]

  model = createModel(ratings);
  recommendations = recommend(model, user_id, ratings.file)
  test_recommendations_index = find(r-> in(r, test_movies[:movie_id]), recommendations[:item])

  result = Dict("test_recommendations" => test_recommendations, "test_noisy_recommendations" => test_noisy_recommendations)
  return result
end

function hypothesys_2(user_id, category, margin, noise_sizes)
  ratings = Recsys.Dataset();

  category_movies = movies[movies[category] .== 1, :movie_id]
  category_movies_ratings_index = find(r-> in(r, category_movies), ratings.file[:item])
  user_ratings_index = find(r-> r == user_id, ratings.file[:user])

  category_user_ratings_index = intersect(category_movies_ratings_index, user_ratings_index)
  category_user_ratings = ratings.file[category_user_ratings_index, :]

  # choose 10% of user ratings in that category ramdomly
  shuffle_index = shuffle([1:size(category_user_ratings, 1)])
  test_ratings_index = category_user_ratings_index[find(r -> r >= size(category_user_ratings, 1) * margin, shuffle_index)]
  test_ratings = ratings.file[test_ratings_index, :]
  sort!(test_ratings, cols = :item)

  predictions_array = []
  mae_array = []
  rmse_array = []

  for i in 1:length(noise_sizes)
    train_ratings = Recsys.Dataset();
    deleterows!(train_ratings.file, test_ratings_index)

    model = createModel(train_ratings);

    popular_movies = get_popular_movies(train_ratings)
    popular_by_category = by_category(popular_movies, category)
    contamine_dataset(train_ratings, user_id, popular_by_category[:item], noise_sizes[i])
    noisy_model = createModel(train_ratings);

    data = ones(Int32, ( size(test_ratings)[1], 2) )
    data[:,1] = data[:,1] * user_id
    data[:,2] = test_ratings[:item]

    items = test_ratings[:item]
    prediction = model.predict(data)
    noisy_prediction = noisy_model.predict(data)
    ratings = test_ratings[:rating]

    result = hcat(items, prediction, noisy_prediction, ratings)

    df = DataFrame()
    df[:item] = int(result[:, 1])
    df[:prediction] = result[:, 2]
    df[:noisy_prediction] = result[:, 3]
    df[:rating] = int(result[:, 4])
    df[:title] = movies[items, :title]

    push!(predictions_array, df)
    push!(mae_array, mae(df[:prediction], df[:noisy_prediction]))
    push!(rmse_array, rmse(df[:prediction], df[:noisy_prediction]))
  end

  result = Dict("noise_mae" => mae_array, "noise_rmse" => rmse_array, "predictions" => predictions_array)
  return result
end

margin = 0.9
user_id = 269
category = :action
noise_sizes = [1, 5, 15]

# Hipótese: Alto impacto ao adicionar um item que o usuário não gosta nas previsões futuras
h1 = hypothesys_1(user_id, category, noise_sizes)

# Hipótese: Adicionando um item de uma categoria que ele não goste irá afetar na previsão dos filmes dessa categoria que ele viu
h2 = hypothesys_2(user_id, category, margin, noise_sizes)
