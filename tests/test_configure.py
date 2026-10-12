import unittest
from unittest.mock import Mock

from base import Ascend, Config, SlurmModel, configure, slurmable


class ConfigureTests(unittest.TestCase):
    def test_child_overrides_preserve_parent_and_sibling_settings(self):
        @configure(framework="ultralytics", account="shared-account", gpus_per_node=2)
        class Framework(SlurmModel):
            pass

        @configure(partition=Ascend.nextgen, time="04:00:00")
        class Child(Framework):
            pass

        @configure(data_version="v2")
        class Sibling(Framework):
            pass

        self.assertEqual(Child.config.framework, "ultralytics")
        self.assertEqual(Child.config.account, "shared-account")
        self.assertEqual(Child.config.gpus, 2)
        self.assertEqual(Child.config.cluster, "ascend")
        self.assertEqual(Child.config.time_limit, "04:00:00")
        self.assertEqual(Framework.config, Config(framework="ultralytics", account="shared-account", gpus=2))
        self.assertEqual(Sibling.config.partition, Framework.config.partition)
        self.assertEqual(Sibling.config.data_version, "v2")
        self.assertIsNot(Child.config, Framework.config)
        self.assertIsNot(Sibling.config, Framework.config)

    def test_explicit_none_clears_inherited_options(self):
        @configure(gpus_per_node=2, cpus_per_task=24, data_file="custom.yaml")
        class Framework(SlurmModel):
            pass

        @configure(cpus_per_task=None, data_file=None)
        class Child(Framework):
            pass

        self.assertEqual(Framework.config.cpus, 24)
        self.assertEqual(Framework.config.data_file, "custom.yaml")
        self.assertEqual(Child.config.cpus, 16)
        self.assertIsNone(Child.config.data_file)

    def test_configuration_preserves_class_identity_and_never_launches(self):
        launch = Mock()
        framework = type("Framework", (SlurmModel,), {"__module__": "__main__", "launch": launch})
        configured = configure(framework="ultralytics")(framework)
        self.assertIs(configured, framework)
        launch.assert_not_called()

    def test_configuration_applies_before_slurmable_launches(self):
        seen = []
        child = type("Child", (SlurmModel,), {
            "__module__": "__main__",
            "launch": classmethod(lambda cls: seen.append(cls.config)),
        })
        decorated = slurmable(configure(partition=Ascend.nextgen, time="04:00:00")(child))
        self.assertIs(decorated, child)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].job_name, "Child")
        self.assertEqual(seen[0].partition, Ascend.nextgen)
        self.assertEqual(seen[0].cluster, "ascend")
        self.assertEqual(seen[0].time_limit, "04:00:00")

    def test_model_without_configuration_inherits_framework_settings(self):
        @configure(framework="ultralytics", time="03:00:00")
        class Framework(SlurmModel):
            pass

        @slurmable
        class Child(Framework):
            pass

        self.assertEqual(Child.config.time_limit, "03:00:00")
        self.assertEqual(Child.config.framework, "ultralytics")
        self.assertEqual(Child.config.job_name, "Child")


if __name__ == "__main__":
    unittest.main()
