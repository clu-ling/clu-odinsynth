from typing import List


class AppInfo:
    """
    General information about the application.

    ***This repo was generated from a cookiecutter template published by myedibleenso and zwellington.
    See https://github.com/clu-ling/clu-template for more info.
    """

    version: str = "0.1-beta"
    description: str = "Python library to manipulate Odinson rules."
    authors: List[str] = ["marcovzla", "myedibleenso", "BeckySharp"]
    contact: str = "gus@parsertongue.org"
    repo: str = "https://github.com/clu-ling/odinson-ruleutils"
    license: str = "Apache 2.0"

    @property
    def download_url(self) -> str:
        return f"{self.repo}/archive/v{self.version}.zip"


info = AppInfo()
