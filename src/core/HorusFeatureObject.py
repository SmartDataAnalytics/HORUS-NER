class HorusFeatureObject:
    #TODO: migrate from horus_matrix to this new object dictionary-based.
    def __init__(self, id_sentence, index_token, token, cv_features=[], tx_features=[]):
        self.id_sentence = id_sentence
        self.index_token = index_token
        self.token = token
        self.cv_features = {}
        self.tx_features = {}
        for fc, value in cv_features:
            self.__addFeature('CV', fc, value)
        for ft, value in tx_features:
            self.__addFeature('TX', ft, value)
        self.pos = ''
        self.ner = ''

    def __addFeature(self, type, feature, value):
        if type=='CV':
            self.cv_features.get(feature,0)
            self.cv_features[feature] = value
        elif type=='TX':
            self.tx_features.get(feature, 0)
            self.tx_features[feature] = value
        else:
            raise Exception('invalid feature type!')

    def __str__(self):
        out =  "id_sentence: {name} \n".format(name=self.id_sentence)
        out += "index token: {name} \n".format(name=self.index_token)
        out += "token: {name} \n".format(name=self.token)

        for cf in self.cv_features.keys():
            value = self.cv_features[cf]
            out += "\t{i}={v}\n".format(i=cf,v=value)

        for cf in self.tx_features.keys():
            value = self.tx_features[cf]
            out += "\t{i}={v}\n".format(i=cf,v=value)
        return out