import importlib
from abc import ABC, abstractmethod
from typing import Callable, Dict, Type

from qiskit.providers import BackendV2, Provider

Filter = Callable[[BackendV2], bool]

class Company(ABC):
    """Class holding qiskit plugin data for a company"""

    installed: Dict[str, "Company"] = {}
    not_installed: Dict[str, Type["Company"]] = {}

    @classmethod
    def get(cls, name: str):
        name = name.lower()
        if name in cls.not_installed:
            company = cls.not_installed[name]

            raise ValueError(
                f"The qiskit plugin for {name} is not installed."
                (
                    f" You can install it with pip install {company.qiskit_module.replace('_', '-')}."
                    if hasattr(company, "qiskit_module") else ""
                )
            )

        return cls.installed[name]

    @classmethod
    def supported_companies(cls):
        return list(cls.installed.keys())

    @property
    @abstractmethod
    def provider_cls(self) -> Type[Provider]:
        """Returns the associated Qiskit Provider class for the company"""

    @classmethod
    def is_plugin_installed(cls) -> bool:
        """Determines if the required python package is installed for the company"""
        return False

    @abstractmethod
    def is_simulator(self, backend: BackendV2) -> bool:
        """Returns if a backend is a simulator"""

def is_simulator(backend: BackendV2):
    """Determines if a backend is a simulator"""
    return any([c.is_simulator(backend) for c in Company.installed.values()])

def register(company_name, module=None):
    """Registers a company with the name"""

    def inner(cls: Type[Company]):
        # If using the module keyword, we can auto-generate a module checking
        # function
        if module:
            cls.qiskit_module = module
            cls.is_plugin_installed = lambda: _check_module(module)

        # Only include companies whose plugins we have installed
        if cls.is_plugin_installed():
            Company.installed[company_name] = cls()
        else:
            Company.not_installed[company_name] = cls

        return cls

    return inner

@register('ibm', module="qiskit_ibm_provider")
class IBM(Company):
    @property
    def provider_cls(self):
        from qiskit_ibm_provider import IBMProvider
        return IBMProvider

    def is_simulator(self, backend: BackendV2):
        return "simulator" in backend.name

@register('ionq', module="qiskit_ionq")
class IonQ(Company):
    @property
    def provider_cls(self):
        from qiskit_ionq import IonQProvider
        return IonQProvider

    def is_simulator(self, backend: BackendV2) -> bool:
        return "simulator" in backend.name

def _check_module(module: str):
    try:
        importlib.import_module(module)
        return True
    except ModuleNotFoundError:
        return False
