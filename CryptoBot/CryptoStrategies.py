import numpy as np

class Strategy:

    def prediction_stat(self, predictions, include_std):
        if len(predictions) > 0:
            prediction = np.mean(predictions) - include_std * np.std(predictions)
        else:
            prediction = 0

        return prediction

    def condition_prediction(self, side, predictions):

        if side == 'bids':
            coeff = -1
        else:
            coeff = 1

        norm_predictions = coeff * predictions
        plus_predictions = norm_predictions[norm_predictions > 0.01]
        minus_predictions = norm_predictions[norm_predictions < -0.01]

        plus_prediction = self.prediction_stat(plus_predictions, 1)
        minus_prediction = self.prediction_stat(minus_predictions, 0)

        return plus_prediction, minus_prediction, coeff


    def determine_move(self, predictions, order_book, portfolio):

        is_holding_usd = portfolio.value['USD'] > 10

        if is_holding_usd:
            side = 'bids'
            opposing_side = 'asks'
            prices = order_book['0'].values
        else:
            side = 'asks'
            opposing_side = 'bids'
            prices = order_book['60'].values

        current_price = prices[-1]
        prediction, opposing_prediction, coeff = self.condition_prediction(side, predictions)
        pred_del = prediction - opposing_prediction

        if (pred_del > 0):
            decision = {'side': side, 'size coeff': 1, 'price': current_price + coeff * prediction}
        elif (pred_del < 0):
            decision = {'side': opposing_side, 'size coeff': 1, 'price': current_price - coeff * 0.01}
        else:
            decision = None

        #decision = {'side': 'bids', 'size coeff': 1, 'price': 128}

        return decision