from typing import Dict, List, Optional, Tuple

from qiskit.providers import BackendV1
from qiskit.providers import BackendV2 as Backend
from qiskit.providers import BackendV2Converter
from qiskit.providers import ProviderV1 as Provider
from qiskit.providers.providerutils import filter_backends

from .company import Company, is_simulator


class MultiPlatformProvider(Provider):
    """Provider instance that manages qiskit providers for
    multiple companies.
    
    Each company's provider will need authentication, which you
    should handle through environment variables.
        - IBM: `QISKIT_IBM_TOKEN`
        - IonQ: `IONQ_API_TOKEN`
    """

    version = 1

    def __init__(self):
        self.companies = Company.installed.copy()

        # Auth should be handled by environment variables
        self.providers: Dict[str, Provider] = {
            name: Company.get(name).provider_cls()
            for name in self.companies
        }

    def backends(
        self,
        name: Optional[str] = None,
        *,
        companies: Optional[str | List[str]] = None,
        simulator_only: bool = False,
        **kwargs
    ) -> List[Backend]:
        """Searches backends for the specified criteria.

        Args:
            name: An optional name to filter with. Specifying a company prefix is required
                (`company:backend`) if `company` is None.

            companies: Filter by company name. Can specify more than one company.

            simulator_only: Restrict the backends to simulators. This can stop you from
                spending your entire budget by accident. Default false.

            **kwargs: Additional filter criteria
        """
        backends: List[Backend] = []
        companies, name = _validate_company_and_backend(companies, name)

        # If the user specifies a company, we can let them search over just that
        if companies is not None:
            for company in companies:
                if company not in self.companies:
                    raise ValueError(
                        f"Unknown company name {company}. "
                        f"Supported companies: {Company.supported_companies()}"
                    )

                provider = self.providers[company]
                backends.extend(provider.backends(name=name, **kwargs))

        # Otherwise retrieve backends from all providers
        else:
            for provider in self.providers.values():
                backends.extend(provider.backends(**kwargs))

        # Convert everything to BackendV2
        for i, backend in enumerate(backends):
            if isinstance(backend, BackendV1):
                backends[i] = BackendV2Converter(backend)

        # To protect your budget, you may filter out all QPUs
        if simulator_only:
            backends = filter_backends(backends, filters=is_simulator)

        return backends

    def get_backend(self, name=None, company=None, **kwargs):
        return super().get_backend(name, company=company, **kwargs)

def _validate_company_and_backend(
    company: Optional[str | List[str]] = None,
    name: Optional[str] = None
) -> Tuple[List[str], Optional[str]]:

    if isinstance(company, str):
        company = [company]

    if name is not None:
        split_name = name.split(':', maxsplit=1)

        if len(split_name) == 1:
            if company is None:
                raise ValueError(
                    "You need to specify the company name with the company field or a prefix, "
                    f"e.g. ionq:ionq_simulator. Got {name}."
                )

        else:
            if company is None:
                company = [split_name[0].lower()]
                name = split_name[1]
            else:
                raise ValueError(
                    "Cannot specify both company and use a prefix in backend name. "
                    f"Got company {company} and name {name}."
                )
    
    if company is not None:
        company = list(map(str.lower, company))

    return company, name
