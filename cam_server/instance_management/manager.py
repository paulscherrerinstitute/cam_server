from logging import getLogger

_logger = getLogger(__name__)


class InstanceManager(object):
    def __init__(self):
        self.instances = {}

    def get_info(self):
        """
        Return the instance manager info.
        :return: Dictionary with the info.
        """
        info = {"active_instances": dict((instance.get_name(), instance.get_info())
                                         for instance in self.instances.values() if instance.is_running())}

        return info

    def add_instance(self, instance_name, instance_wrapper):
        """
        Add instance to the list of instances.
        :param instance_name: Instance name to add.
        :param instance_wrapper: Instance wrapper.
        """
        self.instances[instance_name] = instance_wrapper

    def is_instance_present(self, instance_name):
        """
        Check if instance is already present in the instances pool.
        :param instance_name: Name to check.
        :return: True if instance is already present.
        """
        return instance_name in self.instances

    def get_instance(self, instance_name):
        """
        Retrieve the requested instance.
        :param instance_name: Name od the instance to return.
        :return:
        """
        return self.instances[instance_name]

    def start_instance(self, instance_name):
        """
        Start the instance.
        :param instance_name: Instance to start.
        """
        instance = self.get_instance(instance_name)

        if not instance.is_running():
            instance.start()

        instance.start()

    def stop_instance(self, instance_name):
        """
        Terminate the instance of the specified name.
        :param instance_name: Name of the instance to stop.
        """
        _logger.info("Stopping instance '%s'.", instance_name)

        if instance_name in self.instances:
            self.instances[instance_name].stop()

    def stop_all_instances(self):
        """
        Terminate all the instances.
        :return:
        """
        _logger.info("Stopping all instances.")

        for instance_name in self.instances:
            self.stop_instance(instance_name)
