import os.path

from sphinx.application import Sphinx


class TestDocs:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    source_dir = base_dir
    config_dir = base_dir
    output_dir = os.path.join(base_dir, '_build')
    doctree_dir = os.path.join(base_dir, '_build', 'doctrees')

    def test_html_docs(self):
        app = Sphinx(self.source_dir, self.config_dir, self.output_dir, self.doctree_dir, buildername='html',
                     warningiserror=True)
        app.build(force_all=True)
