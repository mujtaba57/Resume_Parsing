from drf_yasg.utils import swagger_auto_schema
from rest_framework.response import Response
from rest_framework.views import APIView

from Topic.topic import *
from parser_api.serializers import ParserSerializer
from query_extractor.result_extract import *

dictObj = DictParser()


def response_standard(response_data, message, status):
    return {
        "data": response_data,
        "message": message,
        "httpCode": status
    }


class ParserView(APIView):

    @swagger_auto_schema(
        request_body=ParserSerializer,

        responses={
            200: "Parsed",
            401: "Unauthorized",
            500: "Internal Server Error",
        },
    )
    def post(self, request):
        try:
            request_data = request.data
            header = request.headers
            if header['Authorization'] == '6251655368566D597133743677397A24':
                dictObj.input_data(request_data)
                response = dictObj.result_extractor()
                # dictObj.parse_query()
                # result = dictObj.parse_data()
                # json_object = json.dumps(result, indent=4)
                # final_result = dictObj.concatenate_obj(json_object)
                # response = dictObj.reformat_obj(final_result)
                # index_response = dictObj.indexing(response)
                return Response(response_standard(response_data=response, message=None, status=200), '200')
            else:
                return Response(response_standard(response_data=None, message=f"Unauthorized", status=401), '401')
        except Exception as e:
            return Response(response_standard(response_data=None, message=e.args[0], status=500), '500')
