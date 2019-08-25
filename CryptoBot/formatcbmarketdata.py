from cbpro import PublicClient
import numpy as np

def format_data():

    pub_client = PublicClient()
    products = pub_client.get_products()

    data = {}

    for product in products:
        current_dat = {}
        current_dat['base order min'] = float(product['base_min_size'])
        current_dat['base resolution'] = np.abs(int(np.log10(float(product['base_increment']))))
        current_dat['resolution'] = np.abs(int(np.log10(float(product['quote_increment']))))
        current_dat['avoid'] = product['post_only'] or product['limit_only'] or product['cancel_only']
        data[product['id']] = current_dat
        if product['quote_currency'] == 'USD':
            data[product['base_currency']] = current_dat

    print('{')
    for id in data:
        print(" '" + id + "'" + ":" + str(data[id]) + ",")
    print('}')

if __name__ == "__main__":
    format_data()