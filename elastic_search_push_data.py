import os
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import warnings
import requests
import io
import google.generativeai as genai
from PIL import Image
import json
import random
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)

ELASTIC_HOST = "http://localhost:9200"
INDEX_NAME = "products_news"
XLSX_FILE_PATH = "dulieu_1707.xlsx"

try:
    es_client = Elasticsearch(hosts=[ELASTIC_HOST])
    if not es_client.ping():
        raise ConnectionError("Không thể kết nối đến Elasticsearch.")
    print("Kết nối đến Elasticsearch thành công!")
except (ValueError, ConnectionError) as e:
    print(f"Lỗi: {e}")
    exit()

def create_index_with_embedding_mapping():
    """
    Tạo index trong Elasticsearch với mapping mới, bao gồm trường dense_vector.
    """
    if es_client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' đã tồn tại. Xóa index cũ để tạo lại.")
        es_client.indices.delete(index=INDEX_NAME)

    mapping = {
        "properties": {
            "product_code": {"type": "keyword"},
            "product_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "category": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "properties": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "lifecare_price": {"type": "double"},
            "trademark": {"type": "keyword"},
            "guarantee": {"type": "keyword"},
            "inventory": {"type": "integer"},
            "specifications": {"type": "text"},
            "avatar_images": {"type": "keyword"},
            "link_product": {"type": "keyword"},
            "image_embedding": {
                "type": "dense_vector",
                "dims": 512 
            }
        }
    }
    
    print(f"Tạo index mới '{INDEX_NAME}' với mapping cho vector...")
    es_client.indices.create(index=INDEX_NAME, mappings=mapping)
    print("Tạo index thành công.")

def process_and_embed_data():
    """
    Đọc dữ liệu từ XLSX, tải ảnh, tạo embedding và đẩy vào Elasticsearch.
    """
    try:
        df = pd.read_excel(XLSX_FILE_PATH)
        df.columns = [
            'product_code', 'product_name', 'category', 'properties',
            'lifecare_price', 'trademark', 'guarantee', 'inventory',
            'specifications', 'avatar_images', 'link_product'
        ]
        df = df.dropna(subset=['product_code', 'product_name'])
        df['inventory'] = pd.to_numeric(df['inventory'], errors='coerce').fillna(0).astype(int)
        df['lifecare_price'] = pd.to_numeric(df['lifecare_price'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(float)
        df = df.where(pd.notnull(df), None)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{XLSX_FILE_PATH}'.")
        return

    actions = []
    total_rows = len(df)
    API_ENDPOINT = "http://localhost:8000/embed"

    for index, row in df.iterrows():
        print(f"Đang xử lý dòng {index + 1}/{total_rows}: {row['product_name']}")
        
        doc = row.to_dict()
        image_url = row['avatar_images']
        embedding_vector = None

        if isinstance(image_url, str) and image_url.startswith('http'):
            try:
                response = requests.post(API_ENDPOINT, data={"image_url": image_url}, timeout=15)
                response.raise_for_status()
                result = response.json()

                # Kiểm tra có lỗi không
                if "embedding" in result:
                    embedding_vector = result["embedding"]
                    print(" -> Tạo embedding cho ảnh thành công.")
                else:
                    print(" -> Lỗi từ API:", result.get("error", "Không rõ lỗi"))

            except Exception as e:
                print(f" -> Lỗi khi gửi ảnh đến API local: {e}")
        
        doc['image_embedding'] = embedding_vector
        
        action = {
            "_index": INDEX_NAME,
            "_id": row['product_code'],
            "_source": doc
        }
        actions.append(action)

    print(f"\nChuẩn bị index {len(actions)} sản phẩm...")
    try:
        success, failed = bulk(es_client, actions, raise_on_error=False)
        print(f"Index thành công: {success} sản phẩm.")
        if failed:
            print(f"Index thất bại: {len(failed)} sản phẩm.")
            for i, fail_info in enumerate(failed[:5]):
                print(f"  Lỗi {i+1}: {fail_info['index']['error']}")

    except Exception as e:
        print(f"Đã xảy ra lỗi khi index dữ liệu: {e}")

if __name__ == "__main__":
    create_index_with_embedding_mapping()
    process_and_embed_data()
    print("\nQuá trình xử lý và index dữ liệu đã hoàn tất.")