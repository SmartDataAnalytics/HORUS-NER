class MEX(object):
    def __init__(self, id, label, description, author=None, email=None):
        self.id = id
        self.label = label
        self.description = description
        self.author = author
        self.email = email
        self.configurations = []

    def add_configuration(self, configuration):
        self.configurations.append(configuration)

    def __str__(self):
        return str('id=%s, label=%s' % (self.id , self.label))

    __repr__ = __str__

class MEXConfiguration(object):
    def __init__(self, id, horus_enabled, features, dataset_train, dataset_test, dataset_validation=None, cross_validation=1):
        self.id=id
        self.horus_enabled=horus_enabled
        self.features=features
        self.dataset_train=dataset_train
        self.dataset_validation = dataset_validation
        self.dataset_test=dataset_test
        self.cross_validation = cross_validation
        self.executions = []


    def add_execution(self, execution):
        self.executions.append(execution)

    def __str__(self):
        return str('id=%s, horus_enabled?=%s' % (self.id , self.horus_enabled))

    __repr__ = __str__

class MEXExecution(object):
    def __init__(self, id, model, alg, phase, random_state):
        self.id = id
        self.model = model
        self.algorithm = alg
        self.phase = phase
        self.random_state = random_state
        self.performance = []


    def add_performance(self, performance):
        self.performance=performance


    def __str__(self):
        return str('id=%s, model=%s, algorithm=%s, random_state?=%s, phase=%s' % (self.id, self.model,
                                                                         self.algorithm, self.random_state, self.phase))
    __repr__ = __str__

class MEXPerformance(object):
    def __init__(self, klass, p, r, f, a, s):
        self.klass = klass
        self.precision = p
        self.recall = r
        self.f1 = f
        self.accuracy = a
        self.support = s

    def __str__(self):
        return str('class=%s, precision=%s, recall=%s, f1=%s' % (self.klass, self.precision, self.recall, self.f1))

    __repr__ = __str__