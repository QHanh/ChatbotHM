from elastic_search_push_data import es_client, INDEX_NAME
from src.config.settings import PAGE_SIZE

# GỢI Ý: Thêm tham số `require_properties` để linh hoạt hơn trong việc tìm kiếm.
def search_products(product_name: str, category: str = None, properties: str = None, offset: int = 0, require_properties: bool = False) -> list:
    """
    Tìm kiếm sản phẩm trong Elasticsearch bằng cách kết hợp truy vấn,
    ưu tiên (boost) các sản phẩm trùng khớp cao hơn.
    Hỗ trợ phân trang với `offset`.
    """
    if not category:
        category = product_name

    must_clauses = [
        {"match": {"product_name": product_name}},
        {"match": {"category": category}},
    ]
    should_clauses = []

    if properties:
        # GỢI Ý: Nếu `require_properties` là True, thuộc tính sẽ là điều kiện bắt buộc.
        if require_properties:
            must_clauses.append({"match": {"properties": properties}})
        else:
            # Ngược lại, nó chỉ giúp tăng điểm cho kết quả khớp.
            should_clauses.append({"match": {"properties": {"query": properties, "boost": 0.8}}})

    combined_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "bool": {
                            "must": must_clauses,
                            "should": should_clauses,
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