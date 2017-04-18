from __future__ import division

import numpy as np
import pandas as pd



def distance_euclidean(instance1, instance2):
    """Computes the distance between two instances. Instances should be tuples of equal length.
    Returns: Euclidean distance
    Signature: ((attr_1_1, attr_1_2, ...), (attr_2_1, attr_2_2, ...)) -> float"""
    def detect_value_type(attribute):
        """Detects the value type (number or non-number).
        Returns: (value type, value casted as detected type)
        Signature: value -> (str or float type, str or float value)"""
        from numbers import Number
        attribute_type = None
        if isinstance(attribute, Number):
            attribute_type = float
            attribute = float(attribute)
        else:
            attribute_type = str
            attribute = str(attribute)
        return attribute_type, attribute
    # check if instances are of same length
    if len(instance1) != len(instance2):
        raise AttributeError("Instances have different number of arguments."+str(len(instance1))+" "
            +str(len(instance2)))
    # init differences vector
    differences = [0] * len(instance1)
    # compute difference for each attribute and store it to differences vector
    for i, (attr1, attr2) in enumerate(zip(instance1, instance2)):
        type1, attr1 = detect_value_type(attr1)
        type2, attr2 = detect_value_type(attr2)
        # raise error is attributes are not of same data type.
        if type1 != type2:
            raise AttributeError("Instances have different data types.")
        if type1 is float:
            # compute difference for float
            differences[i] = attr1 - attr2
        else:
            # compute difference for string
            if attr1 == attr2:
                differences[i] = 0
            else:
                differences[i] = 1
    # compute RMSE (root mean squared error)
    rmse = (sum(map(lambda x: x**2, differences)) / len(differences))**0.5
    return rmse

class LOF:
    """Helper class for performing LOF computations and instances normalization."""
    def __init__(self, instances, normalize=True, distance_function=distance_euclidean):
        self.instances = instances
        self.normalize = normalize
        self.distance_function = distance_function
        if normalize:
            self.normalize_instances()

        # else:
            # new_instances = [] 
            # for instance in self.instances:
            #     new_instances.append(instance)
            # self.instances = new_instances       

    def compute_instance_attribute_bounds(self):
        feature_len = len(self.instances.first())
        min_values = [float("inf")] * feature_len #n.ones(len(self.instances[0])) * n.inf 
        max_values = [float("-inf")] * feature_len #n.ones(len(self.instances[0])) * -1 * n.inf
        
        for i in range(feature_len):
            xs = instances.map(lambda x: x[i])
            min_values[i] = xs.reduce(lambda x, y: min(x,y))#self.instances.min(lambda x: x[i])
      
            max_values[i] = xs.reduce(lambda x, y: max(x,y))#self.instances.max(lambda x: x[i])

        # for instance in self.instances:
        #     min_values = tuple(map(lambda x,y: min(x,y), min_values,instance)) #n.minimum(min_values, instance)
        #     max_values = tuple(map(lambda x,y: max(x,y), max_values,instance)) #n.maximum(max_values, instance)
        self.max_attribute_values = max_values
        self.min_attribute_values = min_values
        print "max vals: " + str(max_values)
        print "min vals: " + str(max_values)
        
            
    def normalize_instances(self):
        """Normalizes the instances and stores the infromation for rescaling new instances."""
        if not hasattr(self, "max_attribute_values"):
            self.compute_instance_attribute_bounds()

        new_instances = self.instances.map(self.normalize())    
        # new_instances = []
        # for instance in self.instances:
        #     new_instances.append(self.normalize_instance(instance)) # (instance - min_values) / (max_values - min_values)
        self.instances = new_instances
        
    def normalize(self):
        return tuple(map(lambda value,max,min: (value-min)/(max-min) if max-min > 0 else 0, 
                         instance, self.max_attribute_values, self.min_attribute_values))
        

    def normalize_instance(self, instance):
        return tuple(map(lambda value,max,min: (value-min)/(max-min) if max-min > 0 else 0, 
                         instance, self.max_attribute_values, self.min_attribute_values))
        
    def local_outlier_factor(self, min_pts, instance, index=None):
        """The (local) outlier factor of instance captures the degree to which we call instance an outlier.
        min_pts is a parameter that is specifying a minimum number of instances to consider for computing LOF value.
        Returns: local outlier factor
        Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
        if self.normalize:
            instance = self.normalize_instance(instance)
        
        #
        if index != None:
            instances_value_backup = list(self.instances)
            del self.instances[index]
        #

        # return local_outlier_factor(min_pts, instance, self.instances, distance_function=self.distance_function)
        l_o_f = local_outlier_factor(min_pts, instance, self.instances, distance_function=self.distance_function)

        if index != None:       
            self.instances = list(instances_value_backup)
        return l_o_f    



    def get_lrd(index):
        return lrds[index]    

    def estimate_lrds(min_pts, **kwargs):
        lrds = [0] * len(self.instances)
        i = 0
        for instance in self.instances:
            instances = filter(lambda a: not equals(a,instance), self.instances)
            lrds[i] = lrd_estimate(min_pts, instance, instances, **kwargs) 
            i = i + 1   



def k_distance(k, instance, instances, distance_function=distance_euclidean):
    #TODO: implement caching
    """Computes the k-distance of instance as defined in paper. It also gatheres the set of k-distance neighbours.
    Returns: (k-distance, k-distance neighbours)
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> (float, ((attr_j_1, ...),(attr_k_1, ...), ...))"""
    distances = {}
    for instance2 in instances:
        distance_value = distance_function(instance, instance2)
        if distance_value in distances:
            distances[distance_value].append(instance2)
        else:
            distances[distance_value] = [instance2]
    distances = sorted(distances.items())
    neighbours = []
    [neighbours.extend(n[1]) for n in distances[:k]]
    return distances[k - 1][0], neighbours



def reachability_distance(k, instance1, instance2, instances, distance_function=distance_euclidean):
    """The reachability distance of instance1 with respect to instance2.
    Returns: reachability distance
    Signature: (int, (attr_1_1, ...),(attr_2_1, ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(k, instance2, instances, distance_function=distance_function)
    return max([k_distance_value, distance_function(instance1, instance2)])


def lrd_estimate(min_pts, instance, instances, **kwargs):
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    reachability_distances_array = [0]*len(neighbours) #n.zeros(len(neighbours))
    for i, neighbour in enumerate(neighbours):
        reachability_distances_array[i] = reachability_distance(min_pts, instance, neighbour, instances, **kwargs) 
        # print reachability_distances_array[i]
    return len(neighbours) / sum(reachability_distances_array)


def local_reachability_density(min_pts, instance, instances, **kwargs):
    """Local reachability density of instance is the inverse of the average reachability 
    distance based on the min_pts-nearest neighbors of instance.
    Returns: local reachability density
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
    return l.get_lrd(index) #lrd_estimate(min_pts, instance, instances, **kwargs)
# todo index???????


def local_outlier_factor(min_pts, instance, instances, **kwargs):
    """The (local) outlier factor of instance captures the degree to which we call instance an outlier.
    min_pts is a parameter that is specifying a minimum number of instances to consider for computing LOF value.
    Returns: local outlier factor
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)

########
    instance_lrd = lrd_estimate(min_pts, instance, instances, **kwargs)
    lrd_ratios_array = [0]* len(neighbours)
    # print len(instances)
    # print instances.shape

    for i, neighbour in enumerate(neighbours):
        # instances_without_instance = set(instances)
        # instances_without_instance.discard(neighbour)
        instances_without_instance = filter(lambda a: not equals(a,neighbour), instances)

#######
        neighbour_lrd = lrd_estimate(min_pts, neighbour, instances_without_instance, **kwargs)
        lrd_ratios_array[i] = neighbour_lrd / instance_lrd
    return sum(lrd_ratios_array) / len(neighbours)

# def outliers2(k, instances, **kwargs):
#     """Simple procedure to identify outliers in the dataset."""
#     instances_value_backup = instances
#     outliers = []
#     for i, instance in enumerate(instances_value_backup):
#         instances = list(instances_value_backup)
#         instances.remove(instance)
#         # del instances[i]
#         # instances = np.delete(instances, i, axis=0)
#         l = LOF(instances, normalize=True)#**kwargs)
#         value = l.local_outlier_factor(k, instance)
#         if value > 1:
#             outliers.append({"lof": value, "instance": instance, "index": i})
#     outliers.sort(key=lambda o: o["lof"], reverse=True)
#     return outliers

def equals(a,b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False    
    return True        

def lof_single(x, instances_value_backup, k):
    # x = [1,2,3,2,2,2,3,4]
    # filter(lambda a: a != 2, x)

    instances_without_instance = filter(lambda a: not equals(a,x), instances_value_backup) #instances_value_backup.remove(x)
    print len(instances_without_instance)
    # print instances_without_instance
    value = local_outlier_factor(k, x, instances_without_instance ,distance_function=distance_euclidean)
    if value > 1:
        return value#{"lof": value}#, "instance": instance}#, "index": i}

def outliers(k, instances, **kwargs):
    
    instance_number = instances.count()
    print "Instance number = " + str(instance_number)
    l = LOF(instances, normalize=False)
    print "Finished initializing of LOF"
    # l.estimate_lrds(k, **kwargs)
    # print "Finished lrds estimations"
    instances_value_backup = instances.take(instance_number)
    outliers = instances.map(lambda x : lof_single(x,instances_value_backup, k))
    
    # for i, instance in enumerate(instances_value_backup):
    #     print "computing instance " + str(i)
    #     value = l.local_outlier_factor(k, instance,i)
    #     if value > 1:
    #         outliers.append({"lof": value, "instance": instance, "index": i})
    # outliers.sort(key=lambda o: o["lof"], reverse=True)
    return outliers.take(instance_number)



