import abc


class DAFXWrapperBase:
    @abc.abstractmethod
    def get_num_params(self):
        return

    @abc.abstractmethod
    def get_current_param_settings(self):
        return

    @abc.abstractmethod
    def process_effect(self, signal):
        return

    @abc.abstractmethod
    def apply_normalised_index_parameter(self, idx, value):
        return

    @abc.abstractmethod
    def apply_normalised_named_parameter(self, name, value):
        return

    @abc.abstractmethod
    def apply_normalised_parameter_vector(self, vector):
        return

    @abc.abstractmethod
    def apply_unnormalised_index_parameter(self, idx, value):
        return

    @abc.abstractmethod
    def apply_unnormalised_named_parameter(self, name, value):
        return

    @abc.abstractmethod
    def apply_unnormalised_parameter_vector(self, vector):
        return
