#! /usr/bin/env python
##
##This script describes how to generate features and which features were used in this solution.
##

import scipy.stats as stats
import random


def find_corr_pairs_sub(train_x, train_y, eps=0.01):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] - train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                feature_corr_list.append(feature_corr)
    
    return feature_corr_list

def find_corr_pairs_plus(train_x, train_y, eps=0.01):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] + train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, corr[0])
                feature_corr_list.append(feature_corr)
    return feature_corr_list

def find_corr_pairs_mul(train_x, train_y,eps=0.01):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] * train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                feature_corr_list.append(feature_corr)
    
    return feature_corr_list

def find_corr_pairs_divide(train_x, train_y, eps=0.01):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i != j:
                try:
                    res = train_x[:,i] / train_x[:,j]
                    corr = stats.pearsonr(res, train_y)
                    if abs(corr[0]) < eps:
                        continue
                    feature_corr = (i, j, abs(corr[0]))
                    feature_corr_list.append(feature_corr)
                except ValueError:
                    print 'divide 0'
    
    return feature_corr_list

def find_corr_pairs_sub_mul(train_x, train_y, sorted_corr_sub, eps=0.01):
    feature_size = len(train_x[0])
    feature_corr_list = []
    for i in range(len(sorted_corr_sub)):
        ind_i = sorted_corr_sub[i][0]
        ind_j = sorted_corr_sub[i][1]
        if i % 100 == 0:
            print i
        for j in range(feature_size):
            if j != ind_i and j != ind_j :
                res = (train_x[:,ind_i] - train_x[:, ind_j]) * train_x[:,j]
                corr = stats.pearsonr(res, train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (ind_i, ind_j, j, corr[0])
                feature_corr_list.append(feature_corr)
    return feature_corr_list

def get_distinct_feature_pairs(sorted_corr_list):
    distinct_list = []
    dis_ind = {}
    for i in range(len(sorted_corr_list)):
        if sorted_corr_list[i][0] not in dis_ind and sorted_corr_list[i][1] not in dis_ind:
            dis_ind[sorted_corr_list[i][0]] = 1
            dis_ind[sorted_corr_list[i][1]] = 1
            distinct_list.append(sorted_corr_list[i])
    return distinct_list

def get_distinct_feature_pairs2(sorted_corr_list):
    distinct_list = []
    dis_ind = {}
    for sorted_corr in sorted_corr_list:
        cnt = 0
        for i in range(3):
            if sorted_corr[i] in dis_ind:
                cnt = cnt + 1
        if cnt > 1:
            continue
        for i in range(3):
            dis_ind[sorted_corr[i]] = 1
        distinct_list.append(sorted_corr)
    return distinct_list

def get_feature_pair_sub_list(train_x, train_y, eps=0.01):
    sub_list = find_corr_pairs_sub(train_x, train_y, eps)
    sub_list2 = [corr for corr in sub_list if abs(corr[2])>eps]
    sorted_sub_list = sorted(sub_list2, key=lambda corr:abs(corr[2]), reverse=True)
    dist_sub_list = get_distinct_feature_pairs(sorted_sub_list)
    dist_sub_list2 = [[corr[0], corr[1]] for corr in dist_sub_list]
    feature_pair_sub_list = [[520, 521], [271, 521], [271, 520]]
    feature_pair_sub_list.extend(dist_sub_list2[1:])
    return feature_pair_sub_list

def get_feature_pair_plus_list(train_x, train_y, eps=0.01):
    plus_list = find_corr_pairs_plus(train_x, train_y, eps)
    plus_list2 = [corr for corr in plus_list if abs(corr[2])>eps]
    sorted_plus_list = sorted(plus_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_plus_list = get_distinct_feature_pairs(sorted_plus_list)
    feature_pair_plus_list = [[corr[0],corr[1]] for corr in feature_pair_plus_list]
    return feature_pair_plus_list

def get_feature_pair_mul_list(train_x, train_y, eps=0.01):
    mul_list = find_corr_pairs_mul(train_x, train_y, eps)
    mul_list2 = [corr for corr in mul_list if abs(corr[2])>eps]
    sorted_mul_list = sorted(mul_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_mul_list = get_distinct_feature_pairs(sorted_mul_list)
    feature_pair_mul_list = [[corr[0],corr[1]] for corr in feature_pair_mul_list]
    return feature_pair_mul_list

def get_feature_pair_divide_list(train_x, train_y, eps=0.01):
    divide_list = find_corr_pairs_divide(train_x, train_y, eps)
    divide_list2 = [corr for corr in divide_list if abs(corr[2])>eps]
    sorted_divide_list = sorted(divide_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_divide_list = get_distinct_feature_pairs(sorted_divide_list)
    feature_pair_divide_list = [[corr[0],corr[1]] for corr in feature_pair_divide_list]
    return feature_pair_divide_list

def get_feature_pair_sub_mul_list(train_x, train_y, eps=0.01):
    feature_pair_sub_list = get_feature_pair_sub_list(train_x, train_y, eps=0.01)
    sub_mul_list = find_corr_pairs_sub_mul(train_x, train_y, feature_pair_sub_list, eps=0.01)
    sub_mul_list2 = [corr for corr in sub_mul_list if abs(corr[3]) > eps]
    sorted_sub_mul_list = sorted(sub_mul_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_sub_mul_list = get_distinct_feature_pairs2(sorted_sub_mul_list)
    feature_pair_sub_mul_list = [[corr[0], corr[1], corr[2]] for corr in feature_pair_sub_mul_list]
    return feature_pair_sub_mul_list

def get_feature_pair_sub_list_sf(train_x, train_y, eps=0.01):
    #Owing to the features are selected by random sampling, the returned result may be different from what I provide
    sub_list = find_corr_pairs_sub(train_x, train_y, eps)
    sf = random.sample(len(sub_list),500)
    sub_list_sf = [sub_list[i] for i in sf]
    sub_list2 = [[corr[0], corr[1]] for corr in sub_list_sf]
    feature_pair_sub_list_sf = [[520, 521], [271, 521], [271, 520]]
    feature_pair_sub_list_sf.extend(sub_list2[1:])
    return feature_pair_sub_list_sf
