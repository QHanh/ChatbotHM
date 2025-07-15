import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

ELASTIC_HOST = "http://localhost:9200"
INDEX_NAME = "products_news"
XLSX_FILE_PATH = "dulieu_1507.xlsx"

try:
    es_client = Elasticsearch(hosts=[ELASTIC_HOST])
    if not es_client.ping():
        raise ConnectionError("Không thể kết nối đến Elasticsearch.")
    print("Kết nối đến Elasticsearch thành công!")
except ConnectionError as e:
    print(f"Lỗi: {e}")
    print("Hãy chắc chắn rằng bạn đã khởi chạy Elasticsearch.")
    exit()

def create_index_with_mapping():
    """
    Tạo index trong Elasticsearch với mapping cụ thể cho các trường dữ liệu.
    """
    if es_client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' đã tồn tại. Xóa index cũ để tạo lại.")
        es_client.indices.delete(index=INDEX_NAME)

    mapping = {
        "properties": {
            "product_code": {"type": "keyword"},
            "product_name": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "category": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "properties": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "lifecare_price": {"type": "double"},
            "trademark": {"type": "keyword"},
            "guarantee": {"type": "keyword"},
            "inventory": {"type": "integer"},
            "specifications": {"type": "text", "analyzer": "standard"},
            "avatar_images": {"type": "keyword"},
            "link_product": {"type": "keyword"}
        }
    }
    
    print(f"Tạo index mới '{INDEX_NAME}' với mapping...")
    es_client.indices.create(index=INDEX_NAME, mappings=mapping)
    print("Tạo index thành công.")

def index_data_from_xlsx():
    """
    Đọc dữ liệu từ file XLSX và đẩy vào Elasticsearch.
    """
    try:
        df = pd.read_excel(XLSX_FILE_PATH)
        df.columns = [
            'product_code', 'product_name', 'category', 'properties',
            'lifecare_price', 'trademark', 'guarantee', 'inventory',
            'specifications', 'avatar_images', 'link_product'
        ]
        df = df.dropna(how='all')
        df['inventory'] = pd.to_numeric(df['inventory'], errors='coerce').fillna(0).astype(int)
        df['lifecare_price'] = (
            df['lifecare_price'].astype(str)
            .str.replace(',', '')
            .replace('', '0')
            .astype(float)
        )
        df = df.where(pd.notnull(df), None)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{XLSX_FILE_PATH}'.")
        return

    actions = []
    for index, row in df.iterrows():
        doc = {
            'product_code': row['product_code'],
            'product_name': row['product_name'],
            'category': row['category'],
            'properties': row['properties'],
            'lifecare_price': row['lifecare_price'],
            'trademark': row['trademark'],
            'guarantee': row['guarantee'],
            'inventory': row['inventory'],
            'specifications': row['specifications'],
            'avatar_images': row['avatar_images'],
            'link_product': row['link_product']
        }
        action = {
            "_index": INDEX_NAME,
            "_id": row['product_code'],
            "_source": doc
        }
        actions.append(action)

    print(f"Chuẩn bị index {len(actions)} sản phẩm...")
    try:
        success, failed = bulk(es_client, actions)
        print(f"Index thành công: {success} sản phẩm.")
        if failed:
            print(f"Index thất bại: {len(failed)} sản phẩm.")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi index dữ liệu: {e}")

if __name__ == "__main__":
    create_index_with_mapping()
    index_data_from_xlsx()
    print("Phần index dữ liệu đã được định nghĩa. Hãy chạy thủ công nếu cần.")