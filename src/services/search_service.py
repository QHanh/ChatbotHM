import os
from elasticsearch import Elasticsearch
from src.config.settings import PAGE_SIZE
from typing import List, Dict

ELASTIC_HOST = os.environ.get("ELASTIC_HOST", "http://localhost:9200")
INDEX_NAME = os.environ.get("ELASTIC_INDEX", "products_news")

try:
    es_client = Elasticsearch(hosts=[ELASTIC_HOST])
    if not es_client.ping():
        raise ConnectionError("Không thể kết nối đến Elasticsearch từ search_service.")
except ConnectionError as e:
    print(f"Lỗi kết nối trong search_service: {e}")
    es_client = None

def search_products(product_name: str = None, category: str = None, properties: str = None, offset: int = 0, size: int = PAGE_SIZE, strict_properties: bool = False, strict_category: bool = False) -> List[Dict]:
    if not product_name and not category and not properties:
        return []

    body = {
        "query": {
            "bool": {
                "must": [],
                "should": [],
                "filter": []
            }
        },
        "size": size,
        "from": offset
    }

    if product_name:
        # Điều kiện BẮT BUỘC: sản phẩm phải chứa các từ trong tên tìm kiếm
        body["query"]["bool"]["must"].append({
            "match": {
                "product_name": {
                    "query": product_name
                }
            }
        })
        # Điều kiện KHÔNG BẮT BUỘC (để cộng điểm): ưu tiên cao cho khớp chính xác cụm từ
        body["query"]["bool"]["should"].append({
            "match_phrase": {
                "product_name": {
                    "query": product_name,
                    "boost": 10.0
                }
            }
        })

    if category:
        cat_field = "category.keyword" if strict_category else "category"
        if strict_category:
            body["query"]["bool"]["must"].append({"match": {cat_field: category}})
        else:
            # Cộng điểm nếu khớp category, giúp kéo các sản phẩm đúng danh mục lên trên
            body["query"]["bool"]["should"].append({"match": {cat_field: {"query": category, "boost": 5.0}}})

    if properties:
        prop_query = {"match": {"properties": {"query": properties, "operator": "and"}}}
        if strict_properties:
            body["query"]["bool"]["must"].append(prop_query)
        else:
            body["query"]["bool"]["should"].append(prop_query)

    try:
        response = es_client.search(
            index=INDEX_NAME,
            body=body
        )
        hits = [hit['_source'] for hit in response['hits']['hits']]
        print(f"Tìm thấy {len(hits)} sản phẩm (offset={offset}, strict_cat={strict_category}, strict_prop={strict_properties}).")
        return hits
    except Exception as e:
        print(f"Lỗi khi tìm kiếm: {e}")
        return []
    
def search_products_by_image(image_embedding: list, top_k: int = 1, min_similarity: float = 0.97) -> list:
    """
    Thực hiện tìm kiếm k-Nearest Neighbor (kNN) trong Elasticsearch
    để tìm các sản phẩm có ảnh tương đồng nhất.
    Chỉ trả về kết quả nếu độ tương đồng cao hơn một ngưỡng nhất định.
    """
    if not image_embedding:
        return []

    knn_query = {
        "field": "image_embedding", 
        "query_vector": image_embedding,
        "k": top_k,
        "num_candidates": 100 
    }

    try:
        response = es_client.search(
            index=INDEX_NAME,
            knn=knn_query,
            min_score=min_similarity,
            size=top_k,
            _source_includes=[ 
                "product_name", "category", "properties", "specifications", "lifecare_price",
                "inventory", "avatar_images", "link_product"
            ]
        )
        hits = [hit['_source'] for hit in response['hits']['hits']]
        print(f"Tìm thấy {len(hits)} sản phẩm tương đồng (ngưỡng > {min_similarity}).")
        return hits
    except Exception as e:
        print(f"Lỗi khi tìm kiếm bằng vector: {e}")
        return []
