import numpy as np
import openmdao.api as om

class SoftMax(om.ExplicitComponent):
    # soft (smooth) max function by KS aggregation

    def initialize(self):
        self.options.declare('n_pts', types=int, default=1, desc='length of the input vector')
        self.options.declare('rho', types=int, default=10, desc='KS factor')

    def setup(self):
        n_pts = self.options['n_pts']
        arange = np.arange(n_pts)

        self.add_input('f1', shape=(n_pts))
        self.add_input('f2', shape=(n_pts))
        self.add_output('fmax', shape=(n_pts))
        self.declare_partials('*', '*', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        f1 = inputs['f1']
        f2 = inputs['f2']
        rho = self.options['rho']

        tmp = np.exp(rho * f1) + np.exp(rho * f2)
        outputs['fmax'] = np.log(tmp) / rho

    def compute_partials(self, inputs, partials):
        f1 = inputs['f1']
        f2 = inputs['f2']
        rho = self.options['rho']

        tmp = np.exp(rho * f1) + np.exp(rho * f2)
        dtmp_df1 = np.exp(rho * f1) * rho
        dtmp_df2 = np.exp(rho * f2) * rho

        partials['fmax', 'f1'] = 1 / tmp / rho * dtmp_df1
        partials['fmax', 'f2'] = 1 / tmp / rho * dtmp_df2

class AddScalarVectorComp(om.ExplicitComponent):
    # addition of scalar and all components of a vector
    def initialize(self):
        self.options.declare('vec_dim', default=1, desc='length of the input vector')
        self.options.declare('scalar_scale', default=1, desc='scaling factor for the scalar input')
        self.options.declare('vector_scale', default=np.array([1]), desc='scaling factors for the vector input')
        self.options.declare('units', default=None, desc='units of input and output')

    def setup(self):
        n = self.options['vec_dim']
        units = self.options['units']
        scalar_scale = self.options['scalar_scale']
        vector_scale = self.options['vector_scale']

        self.add_input('scalar_input', val=0., units=units)
        self.add_input('vector_input', val=np.zeros(n), units=units)
        self.add_output('scalar_output', val=0., units=units)

        # partials are constant
        self.declare_partials('scalar_output', 'scalar_input', val=scalar_scale)
        self.declare_partials('scalar_output', 'vector_input', rows=np.zeros(n), cols=np.arange(n), val=vector_scale)

    def compute(self, inputs, outputs):
        scalar_scale = self.options['scalar_scale']
        vector_scale = self.options['vector_scale']

        sum_vector = np.sum(inputs['vector_input'] * vector_scale)
        outputs['scalar_output'] = sum_vector + inputs['scalar_input'] * scalar_scale
