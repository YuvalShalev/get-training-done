"""Tests for research server tool logic (arXiv, Kaggle, PWC clients with mocked HTTP)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from gtd.research import arxiv_client, kaggle_client, pwc_client

# ─── Sample response data ────────────────────────────────────────────────────

SAMPLE_ARXIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>42</opensearch:totalResults>
  <entry>
    <title>XGBoost: A Scalable Tree Boosting System</title>
    <summary>Tree boosting is a highly effective machine learning method.</summary>
    <author><name>Tianqi Chen</name></author>
    <author><name>Carlos Guestrin</name></author>
    <link href="http://arxiv.org/abs/1603.02754" rel="alternate" type="text/html"/>
    <published>2016-03-09T00:00:00Z</published>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
    <category term="stat.ML"/>
  </entry>
  <entry>
    <title>LightGBM: A Highly Efficient Gradient Boosting Decision Tree</title>
    <summary>Gradient Boosting Decision Tree is a popular framework.</summary>
    <author><name>Guolin Ke</name></author>
    <link href="http://arxiv.org/abs/1711.08789" rel="alternate" type="text/html"/>
    <published>2017-11-01T00:00:00Z</published>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""

SAMPLE_KAGGLE_DATASETS_JSON = [
    {
        "ref": "uciml/iris",
        "title": "Iris Species",
        "subtitle": "Classify iris plants into three species",
        "totalBytes": 5120,
        "downloadCount": 150000,
    },
    {
        "ref": "hesh97/titanicdataset-traincsv",
        "title": "Titanic Dataset",
        "subtitle": "Training data for Titanic survival prediction",
        "totalBytes": 61194,
        "downloadCount": 80000,
    },
]

SAMPLE_KAGGLE_NOTEBOOKS_JSON = [
    {
        "ref": "alexisbcook/titanic-tutorial",
        "title": "Titanic Tutorial",
        "author": "Alexis Cook",
        "totalVotes": 3200,
        "language": "Python",
    },
    {
        "ref": "startupsci/titanic-data-science-solutions",
        "title": "Titanic Data Science Solutions",
        "author": "Manav Sehgal",
        "totalVotes": 2800,
        "language": "Python",
    },
]

SAMPLE_PWC_JSON = {
    "count": 25,
    "results": [
        {
            "id": "xgboost",
            "title": "XGBoost: A Scalable Tree Boosting System",
            "abstract": "Tree boosting is a highly effective and widely used method.",
            "url_abs": "https://arxiv.org/abs/1603.02754",
            "url_pdf": "https://arxiv.org/pdf/1603.02754",
            "proceeding": "KDD 2016",
            "tasks": [
                {"name": "Classification"},
                {"name": "Regression"},
            ],
        },
        {
            "id": "lightgbm",
            "title": "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
            "abstract": "GBDT is a popular ML approach.",
            "url_abs": "https://arxiv.org/abs/1711.08789",
            "url_pdf": None,
            "proceeding": "NeurIPS 2017",
            "tasks": [{"name": "Tabular Data"}],
        },
    ],
}


# ─── arXiv client tests ──────────────────────────────────────────────────────


class TestSearchArxiv:
    """Tests for the arXiv search client."""

    @patch("gtd.research.arxiv_client.requests.get")
    def test_successful_search(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_ARXIV_XML
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = arxiv_client.search_arxiv("xgboost gradient boosting", max_results=5)

        assert "results" in result
        assert "total_results" in result
        assert result["total_results"] == 42
        assert len(result["results"]) == 2
        assert result["query"] == "xgboost gradient boosting"

    @patch("gtd.research.arxiv_client.requests.get")
    def test_result_structure(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_ARXIV_XML
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = arxiv_client.search_arxiv("xgboost")

        paper = result["results"][0]
        assert paper["title"] == "XGBoost: A Scalable Tree Boosting System"
        assert "Tree boosting" in paper["abstract"]
        assert "Tianqi Chen" in paper["authors"]
        assert "Carlos Guestrin" in paper["authors"]
        assert paper["url"] == "http://arxiv.org/abs/1603.02754"
        assert paper["published"] == "2016-03-09T00:00:00Z"
        assert "cs.LG" in paper["categories"]

    @patch("gtd.research.arxiv_client.requests.get")
    def test_timeout_returns_error(self, mock_get: Mock) -> None:
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = arxiv_client.search_arxiv("test query")

        assert "error" in result
        assert "timed out" in result["error"]
        assert result["query"] == "test query"

    @patch("gtd.research.arxiv_client.requests.get")
    def test_connection_error_returns_error(self, mock_get: Mock) -> None:
        mock_get.side_effect = requests.exceptions.ConnectionError("No connection")

        result = arxiv_client.search_arxiv("test query")

        assert "error" in result
        assert "connect" in result["error"].lower()

    @patch("gtd.research.arxiv_client.requests.get")
    def test_http_error_returns_error(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 500
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        result = arxiv_client.search_arxiv("test query")

        assert "error" in result
        assert "500" in result["error"]

    @patch("gtd.research.arxiv_client.requests.get")
    def test_api_called_with_correct_params(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_ARXIV_XML
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        arxiv_client.search_arxiv("gradient boosting trees", max_results=15)

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["search_query"] == "gradient boosting trees"
        assert call_kwargs[1]["params"]["max_results"] == 15


# ─── Kaggle client tests ─────────────────────────────────────────────────────


class TestSearchKaggleDatasets:
    """Tests for the Kaggle datasets search client."""

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    @patch("gtd.research.kaggle_client.requests.get")
    def test_successful_search(self, mock_get: Mock, mock_auth: Mock) -> None:
        mock_auth.return_value = ("test_user", "test_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_KAGGLE_DATASETS_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = kaggle_client.search_kaggle_datasets("iris", max_results=5)

        assert "results" in result
        assert "total_results" in result
        assert result["total_results"] == 2
        assert result["query"] == "iris"

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    @patch("gtd.research.kaggle_client.requests.get")
    def test_result_structure(self, mock_get: Mock, mock_auth: Mock) -> None:
        mock_auth.return_value = ("test_user", "test_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_KAGGLE_DATASETS_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = kaggle_client.search_kaggle_datasets("iris")

        ds = result["results"][0]
        assert ds["title"] == "Iris Species"
        assert "iris" in ds["description"].lower() or "classify" in ds["description"].lower()
        assert "kaggle.com/datasets/" in ds["url"]
        assert ds["download_count"] == 150000

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    def test_no_credentials_returns_error(self, mock_auth: Mock) -> None:
        mock_auth.return_value = None

        result = kaggle_client.search_kaggle_datasets("iris")

        assert "error" in result
        assert "credentials" in result["error"].lower()

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    @patch("gtd.research.kaggle_client.requests.get")
    def test_timeout_returns_error(self, mock_get: Mock, mock_auth: Mock) -> None:
        mock_auth.return_value = ("test_user", "test_key")
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = kaggle_client.search_kaggle_datasets("iris")

        assert "error" in result
        assert "timed out" in result["error"]

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    @patch("gtd.research.kaggle_client.requests.get")
    def test_auth_failure_returns_error(self, mock_get: Mock, mock_auth: Mock) -> None:
        mock_auth.return_value = ("bad_user", "bad_key")
        mock_response = Mock()
        mock_response.status_code = 401
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        result = kaggle_client.search_kaggle_datasets("iris")

        assert "error" in result
        assert "authentication" in result["error"].lower()


class TestSearchKaggleNotebooks:
    """Tests for the Kaggle notebooks search client."""

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    @patch("gtd.research.kaggle_client.requests.get")
    def test_successful_search(self, mock_get: Mock, mock_auth: Mock) -> None:
        mock_auth.return_value = ("test_user", "test_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_KAGGLE_NOTEBOOKS_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = kaggle_client.search_kaggle_notebooks("titanic", max_results=5)

        assert "results" in result
        assert result["query"] == "titanic"
        assert len(result["results"]) == 2

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    @patch("gtd.research.kaggle_client.requests.get")
    def test_result_structure(self, mock_get: Mock, mock_auth: Mock) -> None:
        mock_auth.return_value = ("test_user", "test_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_KAGGLE_NOTEBOOKS_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = kaggle_client.search_kaggle_notebooks("titanic")

        nb = result["results"][0]
        assert nb["title"] == "Titanic Tutorial"
        assert nb["author"] == "Alexis Cook"
        assert "kaggle.com/code/" in nb["url"]
        assert nb["score"] == 3200
        assert nb["language"] == "Python"

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    @patch("gtd.research.kaggle_client.requests.get")
    def test_sort_by_parameter_passed(self, mock_get: Mock, mock_auth: Mock) -> None:
        mock_auth.return_value = ("test_user", "test_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_KAGGLE_NOTEBOOKS_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        kaggle_client.search_kaggle_notebooks("titanic", sort_by="voteCount")

        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["sortBy"] == "voteCount"

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    def test_no_credentials_returns_error(self, mock_auth: Mock) -> None:
        mock_auth.return_value = None

        result = kaggle_client.search_kaggle_notebooks("titanic")

        assert "error" in result
        assert "credentials" in result["error"].lower()

    @patch("gtd.research.kaggle_client._get_kaggle_auth")
    @patch("gtd.research.kaggle_client.requests.get")
    def test_connection_error_returns_error(self, mock_get: Mock, mock_auth: Mock) -> None:
        mock_auth.return_value = ("test_user", "test_key")
        mock_get.side_effect = requests.exceptions.ConnectionError("No route")

        result = kaggle_client.search_kaggle_notebooks("titanic")

        assert "error" in result
        assert "connect" in result["error"].lower()


# ─── Papers with Code client tests ───────────────────────────────────────────


class TestSearchPapersWithCode:
    """Tests for the Papers with Code search client."""

    @patch("gtd.research.pwc_client.requests.get")
    def test_successful_search(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_PWC_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = pwc_client.search_papers_with_code("gradient boosting", max_results=10)

        assert "results" in result
        assert "total_results" in result
        assert result["total_results"] == 25
        assert len(result["results"]) == 2
        assert result["query"] == "gradient boosting"

    @patch("gtd.research.pwc_client.requests.get")
    def test_result_structure(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_PWC_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = pwc_client.search_papers_with_code("xgboost")

        paper = result["results"][0]
        assert paper["title"] == "XGBoost: A Scalable Tree Boosting System"
        assert "effective" in paper["abstract"]
        assert paper["url"] == "https://arxiv.org/abs/1603.02754"
        assert paper["url_pdf"] == "https://arxiv.org/pdf/1603.02754"
        assert paper["proceeding"] == "KDD 2016"
        assert "Classification" in paper["tasks"]
        assert "Regression" in paper["tasks"]

    @patch("gtd.research.pwc_client.requests.get")
    def test_task_type_filter_sent(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_PWC_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        pwc_client.search_papers_with_code("boosting", task_type="image-classification")

        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["task"] == "image-classification"

    @patch("gtd.research.pwc_client.requests.get")
    def test_task_type_none_not_sent(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_PWC_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        pwc_client.search_papers_with_code("boosting", task_type=None)

        call_kwargs = mock_get.call_args
        assert "task" not in call_kwargs[1]["params"]

    @patch("gtd.research.pwc_client.requests.get")
    def test_timeout_returns_error(self, mock_get: Mock) -> None:
        mock_get.side_effect = requests.exceptions.Timeout("Timed out")

        result = pwc_client.search_papers_with_code("test")

        assert "error" in result
        assert "timed out" in result["error"]

    @patch("gtd.research.pwc_client.requests.get")
    def test_connection_error_returns_error(self, mock_get: Mock) -> None:
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed")

        result = pwc_client.search_papers_with_code("test")

        assert "error" in result
        assert "connect" in result["error"].lower()

    @patch("gtd.research.pwc_client.requests.get")
    def test_http_error_returns_error(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 503
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        result = pwc_client.search_papers_with_code("test")

        assert "error" in result
        assert "503" in result["error"]

    @patch("gtd.research.pwc_client.requests.get")
    def test_null_pdf_url_becomes_none(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_PWC_JSON
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = pwc_client.search_papers_with_code("lightgbm")

        paper_without_pdf = result["results"][1]
        assert paper_without_pdf["url_pdf"] is None
