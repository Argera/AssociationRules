"""
    AUTHOR : Argyrios Theodoridis 2978
                                        """

import pandas as pd
import numpy as np
import keyboard
import time
import copy
import math
import random
import itertools
import csv
import time
import networkx as nx
import cProfile, pstats, io
import matplotlib.pyplot as plt
from pstats import SortKey
from collections import defaultdict

# Reading files and saving the datasets in (pandas dataframe).
user_data = pd.read_csv("ratings.csv")
user_dataframe = pd.DataFrame(user_data, columns = ['userId','movieId','rating','timestamp'])
movie_data = pd.read_csv("movies.csv")
movie_dataframe = pd.DataFrame(movie_data, columns = ['movieId','title','genres'])

print("Minimum rating score of movies in user baskets (give value in [0,5]):", end = " ")
min_score = float(input())


# Create the baskets and fill them with movies with ratings over the rating threshold (min_score).
def CreateMovieBaskets():

    size = user_dataframe.nunique().userId
    user_baskets = [0 for i in range (size-1)]
    user_movies_ids = []
    previous_user = 1

    for ind in user_dataframe.index:
        if (user_dataframe['userId'][ind] == previous_user):
            if (user_dataframe['rating'][ind] >= min_score):
                user_movies_ids.append(user_dataframe['movieId'][ind])

        else:
            user_baskets[previous_user-1] = user_movies_ids
            previous_user = previous_user + 1
            user_movies_ids = []
            user_movies_ids.append(user_dataframe['movieId'][ind])

    return user_baskets


# Creates a dictionary with movies ids as keys and details as values, returns the dictionary.
def ReadMovies():

    movie_details = {}
    temp = {}

    for ind in movie_dataframe.index:
        temp = { movie_dataframe['movieId'][ind] :[movie_dataframe['title'][ind], movie_dataframe['genres'][ind]]}
        movie_details.update(temp)

    return movie_details


# Creates a list with all different movies in dataset
def MovieListMaker():

    counter = 1
    movie_list = []
    temp = {}
    for ind in user_dataframe.index:
        if (user_dataframe['movieId'][ind] not in movie_list):
            movie_list.append(user_dataframe['movieId'][ind])

    return movie_list

# Creates a dictionary with all different movies in dataset
def MovieMapMaker():

    position = 0
    movie_map = {}
    temp = {}

    for i in range(len(movie_list)):
        temp = {position : movie_list[i]}
        movie_map.update(temp)
        position += 1

    return movie_map


# Creates a triangular matrix with all different pairs of movies combinations.
def TriangularMatrixOfPairsCounters():

    n = len(movie_list)
    total_length = int((n*(n-1))/2)
    all_movies_combinations = [0 for i in range(total_length)]

    for k in range(len(user_baskets)):
        for i in range(len(user_baskets[k])):
            for j in range(i,len(user_baskets[k])):
                position = int((i-1) * (n-i/2) + j - i)
                all_movies_combinations[position] += 1

    return all_movies_combinations


# Creates a hash table of counters for pairs of movies,
# as key has pairs of movies and as value a counter of how many times this pair is in dataset.
def HashedCountersOfPairs():

    my_hash_table = {}

    for i in range(len(user_baskets)):
        for subset in itertools.combinations(user_baskets[i], 2):
            if (subset not in my_hash_table):
                my_hash_table[subset] = 1

            else:
                my_hash_table[subset] += 1

    return my_hash_table


# Implements the A-Priori algorithm,
# as inputs takes the baskets aka a list of all different users and for each one all the movies he has seen,
# takes also as input a threshold of frequency that must have every pair if he wants to be called frequent,
# at last one takes the maximum length of movies in the pairs.
def myApriori(items_baskets, min_frequency, max_length):

    N = len(items_baskets)   # The number of different users
    candidate_pairs = []
    total_frequency_pairs = []
    counter = 1
    num_of_pairs = 0

    while (counter <= max_length):
        frequency_items = []
        support = {}

        if (counter == 1):

            for i in range (len(items_baskets)):
                for j in range (len(items_baskets[i])):
                    if (items_baskets[i][j] not in support):
                        support[items_baskets[i][j]] = 1

                    else:
                        support[items_baskets[i][j]] += 1

            for i in support:
                if((support[i] / N) >= min_frequency):
                    frequency_items.append(i)

            frequency_items.sort()
            candidate_pairs = list(itertools.combinations(frequency_items, 2))

        else:
            list_of_pairs = tuple(candidate_pairs)
            temp_candidate_pairs = []
            real_candidate_pairs = []
            my_temp_list = []
            temp_list = set()

            for i in range(len(items_baskets)):
                for j in range(len(list_of_pairs)):
                    temp_counter = 0
                    for k in range (counter):
                        # the next one is pretty slow and if i had the choice, i would made the items_baskets (dictionary) not list
                        if(list_of_pairs[j][k] not in items_baskets[i]):
                            temp_counter = 1
                            break

                    if (temp_counter == 0):
                        if (list_of_pairs[j] not in support):
                            support[list_of_pairs[j]] = 1

                        else:
                            support[list_of_pairs[j]] += 1

            for i in support:
                if(((support.get(i)) / N) >= min_frequency):
                    frequency_items.append(i)

            if not frequency_items:
                print("A-Priori discovered (",num_of_pairs ,") different pairs")
                break;

            frequency_items.sort()

            if (counter >= 2):
                num_of_pairs += len(frequency_items)
                total_frequency_pairs.append(list(frequency_items))

            # Creating possible next ones
            for i in range(len(frequency_items)):
                for j in range(counter):
                    temp_list.add(frequency_items[i][j])

            temp_list = sorted(temp_list)
            temp_candidate_pairs = list(itertools.combinations(temp_list, counter+1))

            ''' The next one works as well as previous but is slower'''
            # for i in range(len(frequency_items)-1):
            #     for j in range(i+1, len(frequency_items)):
            #         for k in range(counter):
            #             temp_list = list(frequency_items[i])
            #             if (frequency_items[j][k] not in temp_list):
            #                 temp_list.append(frequency_items[j][k])
            #                 temp_list.sort()
            #                 if(temp_list not in temp_candidate_pairs):
            #                     temp_candidate_pairs.append(temp_list)

            for i in range (len(temp_candidate_pairs)):
                my_temp_list = list(itertools.combinations(temp_candidate_pairs[i], counter))
                my_temp_list.sort()
                temp_counter = 0

                for j in range(len(my_temp_list)):
                    (list(my_temp_list[j])).sort()
                    if((my_temp_list[j]) not in frequency_items):
                        temp_counter = 1
                        break

                if (temp_counter == 0):
                    real_candidate_pairs.append(tuple(temp_candidate_pairs[i]))

            candidate_pairs = real_candidate_pairs

        counter += 1

        if(counter > max_length):
            print("A-Priori discovered (",num_of_pairs ,") different pairs")

    return total_frequency_pairs


# Application of the APRIORI algorithm, not in its entirety collecting ratings dataset, but in a part of the whole
# Takes shuffled the datas, as input has the length of the part to be processed and returned.
def SampledApriory(number_of_baskets):

    user_data = pd.read_csv("ratings_shuffled.csv")
    user_dataframe = pd.DataFrame(user_data, columns = ['userId','movieId','rating','timestamp'])

    size = user_dataframe.nunique().userId
    baskets_of_all_users = defaultdict(list)
    sample_of_baskets = {}
    real_position = {}
    set_of_users = set()

    for ind in user_dataframe.index:
        current_user = user_dataframe['userId'][ind]
        current_movie = user_dataframe['movieId'][ind]
        baskets_of_all_users[current_user].append(current_movie)

        if (keyboard.is_pressed('s') or keyboard.is_pressed('S')):
            print("break")
            break

        if (user_dataframe['rating'][ind] < min_score):
            continue

        if (current_user not in set_of_users):
            set_of_users.add(current_user)
            if (len(set_of_users) <= number_of_baskets):
                sample_of_baskets[current_user] = baskets_of_all_users[current_user]
                real_position[len(set_of_users)-1] = current_user

            else :
                rand = random.randrange(len(set_of_users))
                if (rand < number_of_baskets):
                    pos = real_position.get(rand)
                    del sample_of_baskets[pos]
                    sample_of_baskets[current_user] = baskets_of_all_users[current_user]
                    real_position[rand] = current_user

        else :
            if (current_user in sample_of_baskets):
                sample_of_baskets[current_user] = baskets_of_all_users[current_user]

    my_list = []
    for i in sample_of_baskets:
        my_list.append(sample_of_baskets.get(i))

    return my_list


# Creates the association rules of frequent itemsets.
# the rules are based in confidence and lift
def AssociationRulesCreation(frequent_itemsets, min_confidence, min_lift, max_lift):

    rules_dictionary = {}
    rules_beyond_confidence = {}
    confidence_of_rules = {}

    for i in range(len(frequent_itemsets)-1,-1,-1):
        length_of_itemset = len(frequent_itemsets[i][0])
        for j in range(len(frequent_itemsets[i])):
            if (length_of_itemset != 2):
                my_temp_list = list(itertools.combinations(frequent_itemsets[i][j],length_of_itemset-1))
                for k in range(length_of_itemset):
                    hypothesis = my_temp_list[k]
                    conclusion = frequent_itemsets[i][j][length_of_itemset-k-1]
                    rules_dictionary [hypothesis] = conclusion

            else:
                hypothesis = frequent_itemsets[i][j][0]
                conclusion = frequent_itemsets[i][j][1]
                rules_dictionary [hypothesis] = conclusion
                rules_dictionary [conclusion] = hypothesis


    for dict in rules_dictionary:
        if(type(dict) == tuple ):
            confidence = ConfidenceCalculation(dict,rules_dictionary.get(dict))
            if (confidence >= min_confidence):
                rules_beyond_confidence[dict] = [rules_dictionary.get(dict)]
                confidence_of_rules[dict] = [confidence]
                len_of_hypothesis = len(dict)

                for i in range(len_of_hypothesis,1,-1):
                    my_temp_list = list(itertools.combinations(dict,len_of_hypothesis-1))
                    for k in range(len(my_temp_list)):
                        conclusion = [rules_dictionary.get(dict)]
                        hypothesis = my_temp_list[k]
                        conclusion.append(dict[len_of_hypothesis-k-1])
                        conclusion = tuple(conclusion)
                        confidence = ConfidenceCalculation(hypothesis,conclusion)

                        if(confidence >= min_confidence):
                            if(hypothesis not in rules_beyond_confidence):
                                rules_beyond_confidence[hypothesis] = [conclusion]
                                confidence_of_rules [hypothesis] = [confidence]

                            else:
                                if (confidence not in rules_beyond_confidence.get(hypothesis)):
                                    rules_beyond_confidence.get(hypothesis).append(conclusion)
                                    confidence_of_rules.get(hypothesis).append(confidence)

        else:
            hypothesis = dict
            conclusion = rules_dictionary.get(dict)
            confidence = ConfidenceCalculation(hypothesis,conclusion)
            if(confidence >= min_confidence):
                if(hypothesis not in rules_beyond_confidence):
                    rules_beyond_confidence[hypothesis] = [conclusion]
                    confidence_of_rules[hypothesis] = [confidence]

                else:
                    if (confidence not in rules_beyond_confidence.get(hypothesis)):
                        rules_beyond_confidence.get(hypothesis).append(conclusion)
                        confidence_of_rules.get(hypothesis).append(confidence)


    conclusion_frequency = {}

    for i in rules_beyond_confidence:
        conclusion = rules_beyond_confidence.get(i)
        for j in range(len(conclusion)):
            if (conclusion[j] not in conclusion_frequency):
                conclusion_frequency[conclusion[j]] = FrequencyOfConclusion(conclusion[j])

    list_of_itemsets = []
    list_of_rules = []
    list_of_hypothesis = []
    list_of_conclusions = []
    list_of_confidences = []
    list_of_lifts = []
    list_of_interests = []
    list_of_rules_ids = []
    id = 1

    for i in rules_beyond_confidence:
        temp_conclusion = rules_beyond_confidence.get(i)
        for j in range (len(temp_conclusion)):
            accepted = 0
            lift = confidence_of_rules.get(i)[j] / conclusion_frequency.get(temp_conclusion[j])
            interest = confidence_of_rules.get(i)[j] - conclusion_frequency.get(temp_conclusion[j])
            if (min_lift > 1):
                if (lift > min_lift):
                    accepted = 1

            if (max_lift > 0 and max_lift < 1):
                if(lift < max_lift):
                    accepted = 1

            if (accepted == 1):
                temp_list = []
                temp_hyp = []
                temp_conc = []
                if (type(i) == tuple):
                    for k in range(len(i)):
                        temp_hyp.append(i[k])
                else:
                    temp_hyp = [i]

                if (type(rules_beyond_confidence.get(i)[j]) == tuple):
                    for k in range (len(rules_beyond_confidence.get(i)[j])):
                        temp_conc.append(rules_beyond_confidence.get(i)[j][k])

                else:
                    temp_conc = [rules_beyond_confidence.get(i)[j]]

                temp_list.extend(temp_hyp)
                temp_list.extend(temp_conc)
                list_of_itemsets.append(temp_list)
                temp_list = "{} --> {}"
                list_of_rules.append(temp_list.format(temp_hyp, temp_conc))
                list_of_hypothesis.append(temp_hyp)
                list_of_conclusions.append(temp_conc)
                list_of_confidences.append(confidence_of_rules.get(i)[j])
                list_of_lifts.append(lift)
                list_of_interests.append(interest)
                list_of_rules_ids.append(id)
                id += 1

    dict = {'itemsets': list_of_itemsets,'hypothesis': list_of_hypothesis,'conclusion':list_of_conclusions,'rule': list_of_rules, 'confidence': list_of_confidences,'lift': list_of_lifts, 'interest': list_of_interests, 'rule ID' : list_of_rules_ids}
    rules_df = pd.DataFrame(dict)

    return rules_df


# Calculates the confidence given the hypothesis and conclusion.
def ConfidenceCalculation(hypothesis,conclusion):

    hypothesis_counter = union_counter = 0

    for i in range (len(user_baskets)):
        hypothesis_flag = conclusion_flag = 0
        if (type(hypothesis) == tuple):
            for j in range(len(hypothesis)):
                if (hypothesis[j] not in user_baskets[i]):
                    hypothesis_flag = 1
                    break

        else:
            if (hypothesis not in user_baskets[i]):
                hypothesis_flag = 1

        if (hypothesis_flag == 0):
            hypothesis_counter += 1
            if (type(conclusion) == tuple):
                for k in range(len(conclusion)):
                    if (conclusion[k] not in user_baskets[i]):
                        conclusion_flag = 1
                        break

            else :
                if (conclusion not in user_baskets[i]):
                    conclusion_flag = 1

            if(conclusion_flag == 0):
                union_counter +=1

    confidence = union_counter / hypothesis_counter

    return confidence


def FrequencyOfConclusion(conclusion):

    conclusion_counter = 0

    for i in range (len(user_baskets)):
        conclusion_flag = 0
        if (type(conclusion) == tuple):
            for j in range(len(conclusion)):
                if (conclusion[j] not in user_baskets[i]):
                    conclusion_flag = 1
                    break
        else:
            if (conclusion not in user_baskets[i]):
                conclusion_flag = 1

        if (conclusion_flag == 0):
            conclusion_counter += 1

    return conclusion_counter / len(user_baskets)


# This is only for presentation
def presentResults(rules_df):

    options_visualization = '''    ===========================================================================
    (a) List ALL discovered rules                                               [format: a]
    (b) List all rules containing a BAG of movies                               [format:
        in their <ITEMSET|HYPOTHESIS|CONCLUSION>                                 b,<i,h,c>,<comma-sep. movie IDs>]
    (c) COMPARE rules with <CONFIDENCE,LIFT>                                    [format: c]
    (h) Print the HISTOGRAM of <CONFIDENCE|LIFT >                               [format: h,<c,l >]
    (m) Show details of a MOVIE                                                 [format: m,<movie ID>]
    (r) Show a particular RULE                                                  [format: r,<rule ID>]
    (s) SORT rules by increasing <CONFIDENCE|LIFT >                             [format: s,<c,l>]
    (v) VISUALIZATION of association rules                                      [format: v,<draw_choice:
        (sorted by lift)                                                         [c(ircular),r(andom),s(pring)]>
    (e) EXIT                                                                    [format: e]
    ---------------------------------------------------------------------------
    Provide your option:'''
    print(options_visualization, end = " ")
    option = str(input())
    option = option.replace(">", "")
    option = option.replace("<", "")
    option = option.split(",")

    if (option[0] is 'a'):
        print(rules_df)
        presentResults(rules_df)

    elif (option[0] is 'b'):
        flag = 0
        itemset = []
        hypothesis = []
        conclusion = []
        ids_of_itemsets = []
        ids_of_hypothesis = []
        ids_of_conclusion = []
        for i in range(1,len(option)):
            if (option[i] is 'i'):
                i+=1

                while ((option[i] is not 'h') and (option[i] is not 'c')):
                    print (option[i])
                    itemset.append(int(option[i]))
                    i+=1
                    if (i == len(option)):
                        flag = 1
                        break

                if flag == 1 :
                    break

            if (option[i] is 'h'):
                i+=1

                while (option[i] is not 'c'):
                    hypothesis.append(int(option[i]))
                    i+=1
                    if (i == len(option)):
                        flag = 1
                        break

                if flag == 1 :
                    break

            if (option[i] is 'c'):
                i+=1

                for j in range(i,len(option)):
                    conclusion.append(int(option[j]))
                break


        if itemset:
            for i in rules_df.index:
                temp_counter = 0
                for j in range(len(itemset)):
                    if (itemset[j] not in rules_df['itemsets'][i]):
                        break
                    else :
                        temp_counter += 1

                if (temp_counter == len(itemset)):
                    ids_of_itemsets.append(rules_df['rule ID'][i])

        if hypothesis:
            for i in rules_df.index:
                temp_counter = 0
                for j in range(len(hypothesis)):
                    if (hypothesis[j] not in rules_df['hypothesis'][i]):
                        break
                    else :
                        temp_counter += 1

                if (temp_counter == len(hypothesis)):
                    ids_of_hypothesis.append(rules_df['rule ID'][i])

        if conclusion:
            for i in rules_df.index:
                temp_counter = 0
                for j in range(len(conclusion)):
                    if (conclusion[j] not in rules_df['conclusion'][i]):
                        break
                    else :
                        temp_counter += 1

                if (temp_counter == len(conclusion)):
                    ids_of_conclusion.append(rules_df['rule ID'][i])

        union = ids_of_itemsets + ids_of_hypothesis + ids_of_conclusion

        for i in union :
            print("\tRule (",i,") :\t",rules_df.iloc[i-1]['rule'])

        presentResults(rules_df)

    elif (option[0] is 'c'):
        fit = np.polyfit(rules_df['lift'], rules_df['confidence'], 1)
        fit_fn = np.poly1d(fit)
        plt.plot(rules_df['lift'], rules_df['confidence'], 'yo', rules_df['lift'],
        fit_fn(rules_df['lift']))
        plt.xlabel('Lift')
        plt.ylabel('Confidence')
        plt.title('CONFIDENCE vs LIFT')
        plt.tight_layout()
        plt.show()
        presentResults(rules_df)

    elif (option[0] is 'h'):
        if (option[1] is 'c'):
            rules_df['confidence'].plot.hist(bins=12, alpha=0.5)
            plt.title('Histogram of CONFIDENCES among discovered rules')
            plt.xlabel('Confidence')
            plt.ylabel('Number of Rules')
            plt.tight_layout()
            plt.show()

        elif (option[1] is 'l'):
            rules_df['lift'].plot.hist(bins=12, alpha=0.5)
            plt.title('Histogram of LIFTS among discovered rules')
            plt.xlabel('Lift')
            plt.ylabel('Number of Rules')
            plt.tight_layout()
            plt.show()

        presentResults(rules_df)

    elif (option[0] is 'm'):

        for i in range(1,len(option)):
            print("\tMovie ID : ",option[i],
                  "\n\tTitle : ",movie_details.get(int(option[i]))[0],
                  "\n\tGenres : ",movie_details.get(int(option[i]))[1],"\n")

        presentResults(rules_df)

    elif (option[0] is 'r'):

        for i in range(1,len(option)):
            print("\n")
            print(rules_df.iloc[i-1])

        presentResults(rules_df)

    elif (option[0] is 's'):
        if(option[1] is 'c'):
            print(rules_df.sort_values('confidence'))

        elif(option[1] is 'l'):
            print(rules_df.sort_values('lift'))

        presentResults(rules_df)

    elif (option[0] is 'v'):
        if(option[1] is 'c'):
            draw_graph(rules_df,'c')

        elif(option[1] is 'r'):
            draw_graph(rules_df,'r')

        elif(option[1] is 's'):
            draw_graph(rules_df,'s')

        presentResults(rules_df)

    elif (option[0] is 'e'):
        print("\n\n\n\n\n\t\t\t\t---------- GOOD BYE ! ----------\n\n\n\n\n")
        return


def draw_graph(rules,draw_choice):

    G = nx.DiGraph()
    color_map = []
    final_node_sizes = []
    color_iter = 0
    NumberOfRandomColors = 100
    edge_colors_iter = np.random.rand(NumberOfRandomColors)
    node_sizes = {}     # larger rule-nodes imply larger confidence
    node_colors = {}    # darker rule-nodes imply larger lift

    for index, row in rules.iterrows():
        if (color_iter < 100):
            color_of_rule = edge_colors_iter[color_iter]
        else:
            continue
        rule = row['rule']
        rule_id = row['rule ID']
        confidence = row['confidence']
        lift = row['lift']
        itemset = row['itemsets']
        hypothesis=row['hypothesis']
        conclusion=row['conclusion']
        G.add_nodes_from(["R"+str(rule_id)])
        node_sizes.update({"R"+str(rule_id): float(confidence)})
        node_colors.update({"R"+str(rule_id): float(lift)})

        for item in hypothesis:
            G.add_edge(str(item), "R"+str(rule_id), color=color_of_rule)

        for item in conclusion:
            G.add_edge("R"+str(rule_id), str(item), color=color_of_rule)

        color_iter += 1 % NumberOfRandomColors

    print("\t++++++++++++++++++++++++++++++++++++++++")
    print("\tNode size & color coding:")
    print("\t----------------------------------------")
    print("\t[Rule-Node Size]")
    print("\t\t5 : lift = max_lilft, 4 : max_lift > lift > 0.75*max_lift + 0.25*min_lift")
    print("\t\t3 : 0.75*max_lift + 0.25*min_lift > lift > 0.5*max_lift + 0.5*min_lift")
    print("\t\t2 : 0.5*max_lift + 0.5*min_lift > lift > 0.25*max_lift + 0.75*min_lift")
    print("\t\t1 : 0.25*max_lift + 0.75*min_lift > lift > min_lift")
    print("\t----------------------------------------")
    print("\t[Rule-Node Color]")
    print("\t\tpurple : conf > 0.9, blue : conf > 0.75, cyan : conf > 0.6, green  : default")
    print("\t----------------------------------------")
    print("\t[Movie-Nodes]")
    print("\t\tSize: 1, Color: yellow")
    print("\t----------------------------------------")

    max_lift = rules['lift'].max()
    min_lift = rules['lift'].min()
    base_node_size = 500

    for node in G:
        if str(node).startswith("R"): # these are the rule-nodes...
            conf = node_sizes[str(node)]
            lift = node_colors[str(node)]
            # rule-node sizes encode lift...
            if lift == max_lift:
                final_node_sizes.append(base_node_size*5*lift)

            elif lift > 0.75*max_lift + 0.25*min_lift:
                final_node_sizes.append(base_node_size*4*lift)

            elif lift > 0.5*max_lift + 0.5*min_lift:
                final_node_sizes.append(base_node_size*3*lift)

            elif lift > 0.25*max_lift + 0.75*min_lift:
                final_node_sizes.append(base_node_size*2*lift)

            else: # lift >= min_lift...
                final_node_sizes.append(base_node_size*lift)

            # rule-node colors encode confidence...
            if conf > 0.9:
                color_map.append('purple')

            elif conf > 0.75:
                color_map.append('blue')

            elif conf > 0.6:
                color_map.append('cyan')

            else: # lift > min_confidence...
                color_map.append('green')

        else: # these are the movie-nodes...
            color_map.append('yellow')
            final_node_sizes.append(2*base_node_size)

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]

    if draw_choice == 'c': #circular layout
        nx.draw_circular(G, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=True)

    elif draw_choice == 'r': #random layout
        nx.draw_random(G, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=True)

    else: #spring layout...
        pos = nx.spring_layout(G, k=16, scale=1)
        nx.draw(G, pos, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=False)
        nx.draw_networkx_labels(G, pos)

    plt.show()

    # discovering most influential and most influenced movies
    # within highest-lift rules...
    outdegree_rules_sequence = {}
    outdegree_movies_sequence = {}
    indegree_rules_sequence = {}
    indegree_movies_sequence = {}

    outdegree_sequence = nx.out_degree_centrality(G)
    indegree_sequence = nx.in_degree_centrality(G)

    for (node, outdegree) in outdegree_sequence.items():
        # Check if this is a rule-node
        if str(node).startswith("R"):
            outdegree_rules_sequence[node] = outdegree
        else:
            outdegree_movies_sequence[node] = outdegree

    for (node, indegree) in indegree_sequence.items():
        # Check if this is a rule-node
        if str(node).startswith("R"):
            indegree_rules_sequence[node] = indegree
        else:
            indegree_movies_sequence[node] = indegree

    max_outdegree_movie_node = max(outdegree_movies_sequence, key=outdegree_movies_sequence.get)
    max_indegree_movie_node = max(indegree_movies_sequence, key=indegree_movies_sequence.get)
    print("\tMost influential movie (i.e., of maximum outdegree) wrt involved rules: ",max_outdegree_movie_node)
    print("\tMost influenced movie (i.e., of maximum indegree) wrt involved rules: ",max_indegree_movie_node)


user_baskets = CreateMovieBaskets()
movie_details = ReadMovies()
movie_list = MovieListMaker()
movie_map = MovieMapMaker()

def main():

    while (True):
        menu = '''        ===========================================================================
        (a) Matrix_of_pairs_counters                                                [format: p]
        (b) Hash_table                                                              [format: h]
        (c) Apriori                                                                 [format: a]
        (d) Sampling                                                                [format: s]
        (e) EXIT                                                                    [format: e]
        ---------------------------------------------------------------------------
        Provide your option:'''
        print(menu,end = " ")
        menu_choice = input()

        if (menu_choice is 'p'):
            matrix_of_pairs_counters = TriangularMatrixOfPairsCounters()
            print(matrix_of_pairs_counters)

        elif (menu_choice is 'h'):
            my_hash_table = HashedCountersOfPairs()
            print(my_hash_table)

        elif (menu_choice is 'a'):
            print("Minimum Frequency(give value in [0 - 1]) :", end =" ")
            min_frequency = float(input())
            print("Maximum length of itemsets (>2) :",end =" ")
            max_length= int(input())
            print("\t\t============== A-PRIORI EXECUTION ============== ")
            start_time = time.time()
            frequent_itemsets = myApriori(user_baskets,min_frequency, max_length)
            print("--- Total time for A-Priori :: %s seconds ---\n" % (time.time() - start_time))
            print("Minimum Confidence(give value in [0.1 - 1]) :", end =" ")
            min_confidence = float(input())
            print("Minimum Lift(give value (>1) or give (-1) to ignore this parameter) :", end =" ")
            min_lift = float(input())
            print("Maximum Lift(give value [0 - 1] or give (-1) to ignore this parameter) :", end =" ")
            max_lift = float(input())
            rules_df = AssociationRulesCreation(frequent_itemsets,min_confidence,min_lift,max_lift)
            print("Input Parameters (min_frequency, min_confidence, min_lift, max_lift, max_length) = [",
            min_frequency, min_confidence, min_lift, max_lift, max_length,"]")
            print("Number of rules = ",len(rules_df))
            presentResults(rules_df)

        elif (menu_choice is 's'):
            print("Before you run the SAMPLING APRIORI you must be super user (you can do that with this command in your terminal)\n----> sudo su\n")
            print("----------------SAMPLING----------------")
            print("Give the sample size of baskets (",int (user_dataframe.nunique().userId/10)," - ",user_dataframe.nunique().userId,") :", end = " ")
            size_of_sampling = int(input())
            sample_of_baskets = SampledApriory(size_of_sampling)
            print("----------------SAMPLING APRIORI----------------")
            print("Minimum Frequency(give value in [0 - 1]) :", end =" ")
            min_frequency = float(input())
            print("Maximum length of itemsets (>2) :",end =" ")
            max_length= int(input())
            print("\t\t============== A-PRIORI EXECUTION ============== ")
            start_time = time.time()
            frequent_itemsets_after_sampling = myApriori(sample_of_baskets,min_frequency, max_length)
            print("--- Total time for A-Priori :: %s seconds ---\n" % (time.time() - start_time))
            print("frequent itemsets after sampling :\n",frequent_itemsets_after_sampling)
            print("For true positives, give the character 'p'")
            character = str(input())
            if ((character is 'p') or (character is 'P')):
                frequent_itemsets = myApriori(user_baskets,min_frequency, max_length)
                true_positives = []
                for k in range(len(frequent_itemsets)):
                    nt1 = map(tuple, frequent_itemsets[k])
                    nt2 = map(tuple, frequent_itemsets_after_sampling[k])
                    st1 = set(frequent_itemsets[k])
                    st2 = set(frequent_itemsets_after_sampling[k])
                    true_positives.append(list(st1.intersection(st2)))

                print("True positives :\n",true_positives)

            # false_positives = []
            # for i in range (len(frequent_itemsets_after_sampling)):
            #     false_positives.append(np.array(frequent_itemsets_after_sampling[i]) - np.array(true_positives[i]))
            #
            # false_negatives = []
            # for i in range (len(frequent_itemsets)):
            #     false_negatives.append(np.array(frequency_items) - np.array(frequent_itemsets_after_sampling))
            #
            # if (f_n < 0):
            #     f_n = 0
            #
            # t_p = len(true_positives)
            # f_p = len(false_positives)
            # f_n = len(false_negatives)
            # precision = t_p / (t_p + f_p)
            # recall = t_p / (t_p + f_n)
            # f1_score = 2*recall*precision /(recall + precision)
            # print ("f1-score = ",f1_score)

        elif (menu_choice is 'e'):
            print("\n\n\n\n\n\t\t\t\t---------- GOOD BYE ! ----------\n\n\n\n\n")
            return


main()
