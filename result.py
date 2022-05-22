from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import logging

class HtmlResult():
    def __init__(self, results):
        self.theme = "themes/result.html"
        self.results = results

    def as_html(self):
        with open(self.theme) as f:
            html = f.read()

        rows = []
        for result in self.results:
            result.as_logs()
            row = '''
                <li class="table-row">
                    <div class="col col-1">{}</div>
                    <div class="col col-2">{:.4f}</div>
                    <div class="col col-3">{:.4f}</div>
                    <div class="col col-4">{:.4f}</div>
                    <div class="col col-5">{:.4f}</div>
                </li>
            '''.format(result.label, result.accuracy, result.precision, result.recall, result.f1_score)
            rows.append(row)

        if len(rows) == 0:
            raise ValueError("No results to display.")

        return html.replace("{result}", "\n".join(rows))

    def write_html(self, path):
        with open(path, "w") as f:
            f.write(self.as_html())


class Result():
    def __init__(self, y_true, y_pred, label):
        self.y_pred = y_pred
        self.y_test = y_true
        self.recall = recall_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.f1_score = f1_score(y_true, y_pred)
        self.accuracy = accuracy_score(y_true, y_pred)
        self.label = label

    def as_logs(self):
        logging.info("Validation of %s model: Precision is: %s Recall is: %s f1 is: %s accuracy is: %s",
                     self.label, self.precision, self.recall, self.f1_score, self.accuracy)