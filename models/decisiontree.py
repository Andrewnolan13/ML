class PredictionDataType:pass
class TrainingDataTypes:pass

def answer_question(question:tuple,feature_row:list)->bool:
    '''
    dim_feature,operation,value = 0, '==', 'Green'
    dim_feature,operation,value = 1, '<=', 1
    '''
    dim_feature,operation,value = question
    if operation == '==':
        return feature_row[dim_feature] == value
    else:
        return feature_row[dim_feature] <= value


class Node:
    def __init__(self,features:list,labels:list,parent = None):
        self.features = features
        self.labels = labels
        self.count = len(labels)
        self.impurity = self.get_impurity()
        self.isLeaf:bool = False
        self.parent = parent
        self.question = None

    def __repr__(self)->str:
        repr =  f"X = {self.features}\n"
        repr += f"y = {self.labels}\n"
        repr += f"count = {self.count}\n"
        repr += f"impurity = {self.impurity}\n"
        repr += f"isLeaf : {self.isLeaf}\n"
        repr += f"Question : {self.question}" if self.question else ''
        return repr

    def get_impurity(self)->float:
        gini_number = 1
        for label in set(self.labels):
            gini_number -= (self.labels.count(label)/self.count)**2
        return gini_number

    def _midpoints(self,col:set)->list[float]:
        distances = sorted(list(col))
        distances = [(distances[i+1]+distances[i])/2 for i in range(len(distances)-1)] + [distances[-1]+1]
        return distances
        
    def generate_questions(self)->list[tuple]:
        dim_features:int = self.features[0].__len__()
        questions:list = list()

        for dim in range(dim_features):
            col = set([row[dim] for row in self.features])
            dtype = self.features[0][dim]
            if isinstance(dtype,(str,bool)):
                questions += [(dim,'==',val) for val in col]
            else:
                questions += [(dim,'<=',val) for val in self._midpoints(col)]

        return questions

    def split_nodes(self,question:tuple)->tuple['Node','Node']:
        '''
        returns two nodes, left node with answer being false and right node with answer being true 
        '''
        false_features,true_features,false_labels,true_labels = [],[],[],[]
        for feature_row,label in zip(self.features,self.labels):
            if answer_question(question,feature_row):
                true_features.append(feature_row)
                true_labels.append(label)
            else:
                false_features.append(feature_row)
                false_labels.append(label)
        return Node(false_features,false_labels,parent = self),Node(true_features,true_labels,parent = self)
    
    def generate_children(self)->None:
        questions:list[tuple] = self.generate_questions()
        max_information_gain:float = 0

        for question in questions:
            left_node,right_node = self.split_nodes(question)
            if left_node.count == 0 or right_node.count == 0:
                continue
            
            information_gain = self.impurity-(left_node.impurity*left_node.count+right_node.impurity*right_node.count)/self.count

            if information_gain > max_information_gain:
                max_information_gain = information_gain
                left_child:Node = left_node
                right_child:Node = right_node
                best_question:str = question
        
        
        if max_information_gain>0:
            self.children:list = [left_child,right_child]
            self.question = best_question
            left_child.generate_children()
            right_child.generate_children()
        else:
            index_of_self:int = self.parent.children.index(self)
            self.parent.children[index_of_self] = Leaf(self)

class Leaf(Node):
    def __init__(self,bud:Node):
        super().__init__(bud.features,bud.labels,parent = bud.parent)
        self.isLeaf = True
    
    def get_probabilities(self)->dict:
        res = dict()
        for label in set(self.labels):
            res[label] = self.labels.count(label)/self.count
        return res
    
    def predict(self)->PredictionDataType:
        probabilities = self.get_probabilities()
        inverse_dict = dict(zip(probabilities.values(),probabilities.keys()))
        return inverse_dict[max(inverse_dict)] 
    
class DecisionTree:
    '''
    Fits N-Dimensional input data to 1-Dimensional target data 
    '''
    def __init__(self,features:list[list],labels:list):
        self.features = features
        self.labels = labels
        self.root = Node(features,labels,parent = self)
        self.children = [self.root]

    def fit(self):
        self.root.generate_children()
        self.depth = self._getDepth(self.root)
    
    def predict(self,feature_row:list,probabilities:bool = False)->PredictionDataType:
        node = self.root

        # idea is to ask the feature questions until I hit a leaf
        while True:
            if isinstance(node,Leaf):
                break
            question = node.question
            if answer_question(question,feature_row):#go right
                node = node.children[1]
            else:
                node = node.children[0]
        
        return node.get_probabilities() if probabilities else node.predict() 
    
    def predict_training_set(self,probabilities:bool = False)->list[list[TrainingDataTypes,PredictionDataType]]:
        res = list()
        for feature_row in self.features:
            pred = self.predict(feature_row,probabilities)
            res.append([*feature_row,pred])
        return res
    
    def visualize(self,columns:list[str] = None):
        if columns is None:
            columns = list(range(self.features[0].__len__()))
        def _visualize_node(node, depth=0):
            indent = "  " * depth
            if isinstance(node, Leaf):
                print(f"{indent}Leaf: {node.predict()}")
            else:
                col = node.question[0]
                print(f"{indent}Question: {columns[col]}{node.question[1]}{node.question[2]}")
                _visualize_node(node.children[0], depth + 1)
                _visualize_node(node.children[1], depth + 1)

        _visualize_node(self.root)

    def _getDepth(self,node,depth=-1):
        if isinstance(node,Leaf):
            return depth
        else:
            return max([self._getDepth(child,depth+1) for child in node.children])    
    
if __name__ == '__main__':
    data = [
        ['Green'    ,3,'Apple'],
        ['Yellow'   ,3,'Apple'],
        ['Red'      ,1,'Grape'],
        ['Red'      ,1,'Grape'],
        ['Yellow'   ,3,'Lemon'],
    ]

    X = [row[:2] for row in data]
    y = [row[2] for row in data]

    tree = DecisionTree(X,y)
    tree.fit()
    tree.predict_training_set()

    accuracy = tree.predict_training_set()
    print(sum([row[2] == datarow[2] for row,datarow in zip(accuracy,data)])/len(data))