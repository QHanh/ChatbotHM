from elastic_search_push_data import es_client, INDEX_NAME
from src.config.settings import PAGE_SIZE

def search_products(
    product_name: str,
    category: str = None,
    properties: str = None,
    offset: int = 0,
    strict_properties: bool = False,
    strict_category: bool = False
) -> list:
    """
    Tìm kiếm sản phẩm trong Elasticsearch.
    Hỗ trợ tìm kiếm cân bằng, không quá rộng cũng không quá nghiêm ngặt.
    """
    if not category:
        category = product_name

    must_clauses = []
    should_clauses = []
    filter_clauses = [] # Filter không dùng trong logic này nữa để linh hoạt hơn

    # Tên sản phẩm luôn là điều kiện quan trọng
    must_clauses.append({"match": {"product_name": {"query": product_name, "boost": 2.0}}})

    # GỢI Ý: Logic mới cân bằng hơn cho category
    if category:
        if strict_category:
            # Khi xem thêm, BẮT BUỘC phải khớp category, nhưng dùng 'match' để linh hoạt
            must_clauses.append({"match": {"category": category}})
        else:
            # Khi tìm lần đầu, category là yếu tố "nên có" để tăng điểm
            should_clauses.append({"match": {"category": category}})

    if properties:
        if strict_properties:
            must_clauses.append({"match": {"properties": properties}})
        else:
            should_clauses.append({"match": {"properties": {"query": properties, "boost": 0.8}}})

    combined_query = {
        "query": {
            "bool": {
                "must": must_clauses,
                "should": should_clauses,
                "filter": filter_clauses, # Sẽ rỗng, nhưng giữ lại cấu trúc
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
        print(f"Tìm thấy {len(hits)} sản phẩm (offset={offset}, strict_cat={strict_category}, strict_prop={strict_properties}).")
        return hits
    except Exception as e:
        print(f"Lỗi khi tìm kiếm: {e}")
        return []