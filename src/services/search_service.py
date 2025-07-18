from elastic_search_push_data import es_client, INDEX_NAME
from src.config.settings import PAGE_SIZE

def search_products(product_name: str, category: str = None, properties: str = None, offset: int = 0) -> list:
    """
    Tìm kiếm sản phẩm trong Elasticsearch bằng cách kết hợp truy vấn,
    ưu tiên (boost) các sản phẩm trùng khớp cao hơn.
    Hỗ trợ phân trang với `offset`.
    """
    if not category:
        category = product_name

    combined_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "bool": {
                            "must": [
                                {"match": {"product_name": product_name}},
                                {"match": {"category": category}},
                                {"match": {"properties": properties}}
                            ],
                            "boost": 3
                        }
                    },
                    {
                        "multi_match": {
                            "query": product_name,
                            "fields": [
                                "product_name^2",
                                "category",
                                "specifications",
                                "trademark"
                            ],
                            "fuzziness": "AUTO"
                        }
                    }
                ]
            }
        }
    }
    try:
        response = es_client.search(
            index=INDEX_NAME, 
            body=combined_query, 
            size=PAGE_SIZE,
            from_=offset
        )
        hits = [hit['_source'] for hit in response['hits']['hits']]
        print(f"Tìm thấy {len(hits)} sản phẩm với truy vấn kết hợp (offset={offset}).")
        return hits
    except Exception as e:
        print(f"Lỗi khi tìm kiếm kết hợp: {e}")
        return []