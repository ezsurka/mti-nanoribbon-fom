# Standard library imports
import functools
import inspect
from pathlib import Path

# External package imports
import scipy.sparse.linalg as sla
import scipy.sparse as sp
import adaptive
import kwant

# Internal imports


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer


class QuickAdptive():
    def __init__(self, params_function, learner_bounds_names,
                 fname_prefix='data_learner_', arg_picker=None, **kwargs):
        self.fname_prefix = fname_prefix
        self.f_args = inspect.getfullargspec(params_function).args
        self.runner = None
        #print(self.f_args)
        #print(kwargs.keys())
        if not set(self.f_args).issubset(set(kwargs.keys())):
            raise Exception(
                "The learning function arguments are not a subset"
                "of the specified parameters missing parameters: %s" %
                self.f_args - kwargs.keys()
            )

        self.combos = kwargs.copy()
        bounds = []
        for learner_bounds_name in learner_bounds_names:
            del self.combos[learner_bounds_name]
            bounds.append(kwargs[learner_bounds_name])

        def function1D(X, *args, **kwargs):
            learner_params = {learner_bounds_names[0]: X}
            return params_function(**learner_params, **kwargs)

        def functionND(X, *args, **kwargs):
            learner_params = {learner_bounds_names[i]:
                              x for i, x in enumerate(X)}
            return params_function(**learner_params, **kwargs)

        if len(learner_bounds_names) == 1:
            Learner = adaptive.Learner1D
            bounds = bounds[0]
            function = function1D
        elif len(learner_bounds_names) == 2:
            Learner = adaptive.Learner2D
            function = functionND
        else:
            Learner = adaptive.LearnerND
            function = functionND

        if arg_picker:
            Learner = adaptive.make_datasaver(
                Learner,
                arg_picker=arg_picker
            )

        self.learner = adaptive.BalancingLearner.from_product(
            function,
            Learner,
            dict(bounds=bounds),
            self.combos
        )

    def runn(self, goal,  executor, periodic_saving_folder=None, interval=100):
        def all_goal(bl):
            return all([goal(learner) for learner in bl.learners])

        if self.runner is None:
            print("Start runner")
            self.runner = adaptive.Runner(
                self.learner,
                executor=executor,
                goal=goal
            )
        else:
            if self.runner.task.done():
                print("The runner is still running so no new one was started.")
            else:
                print("Runner faild start new runner")
                self.runner = adaptive.Runner(
                    self.learner,
                    executor=executor,
                    goal=all_goal
                )

        if periodic_saving_folder:
            combo_fname = self.create_combo_fname(periodic_saving_folder)
            task = self.runner.start_periodic_saving(
                dict(fname=combo_fname), interval)
            print(task)

        return self.runner.task#.live_info()

    def create_combo_fname(self, folder):
        folder = Path(folder)

        def combo_fname(learner):
            keywords = learner.function.keywords
            fname_params = []
            for k, v in keywords.items():
                if isinstance(v, float):
                    fname_params.append(f'{k}{v:3.4f}')
                else:
                    fname_params.append(f'{k}{v}')

            fname = '_'.join(fname_params)
            fname = self.fname_prefix + fname.replace('.', 'p') + '.pickle'
            return folder / fname
        return combo_fname

    def save(self, folder):
        combo_fname = self.create_combo_fname(folder)
        self.learner.save(combo_fname)

    def load(self, folder):
        combo_fname = self.create_combo_fname(folder)
        self.learner.load(combo_fname)
        print(len(self.learner.data.keys()))

    def mapping(self):
        cdims = self.learner._cdims_default
        mapping = {
            tuple(_cdims.values()):
            learner for learner, _cdims in zip(self.learner.learners, cdims)
        }
        return mapping


def mumps_eigsh(A, k, sigma, **kwargs):
    """Call sla.eigsh with mumps support.
    Please see scipy.sparse.linalg.eigsh for documentation.
    """
    class LuInv(sla.LinearOperator):

        def __init__(self, A):
            instance = kwant.linalg.mumps.MUMPSContext()
            instance.analyze(A, ordering='pord')
            instance.factor(A)
            self.solve = instance.solve
            sla.LinearOperator.__init__(self, A.dtype, A.shape)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    opinv = LuInv(A - sigma * sp.identity(A.shape[0]))
    return sla.eigsh(A, k, sigma=sigma, OPinv=opinv, **kwargs)
